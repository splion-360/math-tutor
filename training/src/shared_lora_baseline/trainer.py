from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol, cast

from shared_lora_baseline.config import TrainingConfig
from shared_lora_baseline.constants import FROZEN_MODEL_ID, FROZEN_MODEL_REVISION
from shared_lora_baseline.dry_run import (
    RunPlan,
    add_runtime_versions,
    add_tracking_metadata,
    build_run_plan,
    write_run_metadata,
)
from shared_lora_baseline.gradient_probe import (
    gradient_probe_metrics,
    measure_lora_layer_gradient_energy,
)
from shared_lora_baseline.tracking import start_experiment_tracking
from shared_lora_baseline.validation import load_training_records

SYSTEM_PROMPT = (
    "You generate concise, runnable Manim Community Edition Python scenes for math tutoring. "
    "Return only Python code."
)


class ChatTemplateTokenizer(Protocol):
    eos_token: str | None
    eos_token_id: int | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


def train_shared_lora(config: TrainingConfig) -> RunPlan:
    plan = build_run_plan(config)
    tracking_run = start_experiment_tracking(config, plan)
    plan = add_tracking_metadata(plan, tracking_run.metadata)
    write_run_metadata(plan, config.metadata_path)

    try:
        # Heavy training libraries are imported only after cheap validation succeeds.
        datasets = import_module("datasets")
        peft = import_module("peft")
        torch = import_module("torch")
        transformers = import_module("transformers")
        transformers.set_seed(config.seed)

        records = load_training_records(config.train_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            FROZEN_MODEL_ID,
            revision=FROZEN_MODEL_REVISION,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = (
            transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if config.load_in_4bit
            else None
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            FROZEN_MODEL_ID,
            revision=FROZEN_MODEL_REVISION,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if config.load_in_4bit:
            model = peft.prepare_model_for_kbit_training(model)

        lora_config = peft.LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            task_type=peft.TaskType.CAUSAL_LM,
        )
        model = peft.get_peft_model(model, lora_config)
        trainable_parameters, total_parameters = model.get_nb_trainable_parameters()

        dataset = datasets.Dataset.from_list(
            [_format_record(record, tokenizer) for record in records]
        )

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            encoded = cast(
                dict[str, Any],
                tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=config.max_seq_length - 1,
                    padding=False,
                ),
            )
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None:
                for index, input_ids in enumerate(encoded["input_ids"]):
                    if not input_ids or input_ids[-1] != eos_token_id:
                        input_ids.append(eos_token_id)
                        if "attention_mask" in encoded:
                            encoded["attention_mask"][index].append(1)
            return encoded

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        training_args = transformers.TrainingArguments(
            output_dir=str(config.output_dir),
            seed=config.seed,
            max_steps=config.max_steps,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            logging_steps=1,
            save_steps=config.max_steps,
            save_total_limit=1,
            report_to=tracking_run.report_to,
        )
        collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=collator,
        )
        if config.layer_energy_probe_top_k > 0:
            probe = measure_lora_layer_gradient_energy(
                model=model,
                tokenized_dataset=tokenized,
                data_collator=collator,
                sample_count=config.layer_energy_probe_sample_count,
                top_k=config.layer_energy_probe_top_k,
            )
            plan.metadata["layer_energy_probe"] = probe
            tracking_run.log_metrics(gradient_probe_metrics(probe))
        train_output = trainer.train()
        tracking_run.log_metrics(getattr(train_output, "metrics", {}))
        if config.run_smoke_eval:
            eval_metrics = cast(dict[str, object], trainer.evaluate(eval_dataset=tokenized))
            tracking_run.log_metrics(eval_metrics)
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        plan = add_runtime_versions(
            plan,
            {
                "accelerate": _distribution_version("accelerate"),
                "bitsandbytes": _distribution_version("bitsandbytes"),
                "datasets": str(datasets.__version__),
                "peft": str(peft.__version__),
                "safetensors": _distribution_version("safetensors"),
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
            },
        )
        plan.metadata["parameter_budget"] = {
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_percent": round(100 * trainable_parameters / total_parameters, 6),
        }
        if config.run_smoke_eval:
            plan.metadata["smoke_eval"] = {
                "dataset": "tokenized_training_fixture",
                "purpose": "modal_trainer_smoke",
            }
        plan = add_tracking_metadata(plan, tracking_run.metadata)
        write_run_metadata(plan, config.metadata_path)
        return plan
    finally:
        tracking_run.finish()


def _format_record(
    record: dict[str, Any], tokenizer: ChatTemplateTokenizer
) -> dict[str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Difficulty: {record['difficulty']}\n"
                f"Topic: {record['topic']}\n"
                f"Task: {record['prompt']}"
            ),
        },
        {"role": "assistant", "content": str(record["manim_code"])},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if tokenizer.eos_token is not None and not text.rstrip().endswith(tokenizer.eos_token):
        text = text.rstrip() + tokenizer.eos_token
    return {"text": text}


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"
