"""Run the shared LoRA training workflow for Qwen Manim generation.
This module orchestrates validation, model setup, tracking, probes, training, and metadata."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

from dynamic_lora.config import TrainingConfig
from dynamic_lora.constants import FROZEN_MODEL_ID, FROZEN_MODEL_REVISION
from dynamic_lora.data import (
    format_training_record,
    load_training_records,
    tokenize_training_batch,
)
from dynamic_lora.fixed_prompt_signatures import (
    ProbeExample,
    build_fixed_prompt_signature_callback,
    select_probe_indices,
)
from dynamic_lora.gradient_signatures import (
    GradientSignatureCallback,
    build_gradient_signature_callback,
)
from dynamic_lora.layer_selection import (
    gradient_probe_metrics,
    measure_lora_layer_gradient_energy,
)
from dynamic_lora.run_plan import (
    RunPlan,
    add_runtime_versions,
    add_tracking_metadata,
    build_run_plan,
    write_run_metadata,
)
from dynamic_lora.signature_selection import select_base_probe_modules
from dynamic_lora.tracking import start_experiment_tracking


def train_shared_lora(config: TrainingConfig) -> RunPlan:
    plan = build_run_plan(config)
    reference_layers: tuple[str, ...] = ()
    if config.signature_probe_metadata_path is not None:
        reference_layers = select_base_probe_modules(
            config.signature_probe_metadata_path,
            train_hash=plan.metadata["dataset"]["training_content_sha256"],
            holdout_hash=plan.metadata["dataset"]["holdout_content_sha256"],
            max_seq_length=config.max_seq_length,
            seed=config.seed,
            target_modules=config.target_modules,
        )
        plan.metadata["signature_selection"] = {
            "source": "no_update_base_weight_probe",
            "selected_modules": list(reference_layers),
            "reference_path": str(config.signature_probe_metadata_path),
        }
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
        probe_indices: tuple[int, ...] = ()
        training_indices: tuple[int, ...] = tuple(range(len(records)))
        if config.fixed_prompt_probe_sample_count > 0:
            probe_indices, training_indices = select_probe_indices(
                records,
                sample_count=config.fixed_prompt_probe_sample_count,
                seed=config.seed,
            )
            plan.metadata["fixed_prompt_probe_selection"] = {
                "sample_ids": [records[index]["id"] for index in probe_indices],
                "training_record_count": len(training_indices),
                "selection": "seeded_sha256_of_record_id",
            }
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            FROZEN_MODEL_ID,
            revision=FROZEN_MODEL_REVISION,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = datasets.Dataset.from_list(
            [format_training_record(record, tokenizer) for record in records]
        )

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return tokenize_training_batch(
                batch, tokenizer, max_seq_length=config.max_seq_length
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        training_dataset = (
            tokenized.select(list(training_indices)) if probe_indices else tokenized
        )
        probe_examples = tuple(
            ProbeExample(record_id=records[index]["id"], tokenized=tokenized[index])
            for index in probe_indices
        )

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
            train_dataset=training_dataset,
            data_collator=collator,
        )
        signature_callback: GradientSignatureCallback | None = None
        selected_layers = reference_layers
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
            selected_layers = tuple(probe.get("selected_layers", ()))
        if config.gradient_signature_dim > 0:
            if not selected_layers:
                raise RuntimeError(
                    "gradient signatures were requested but no LoRA modules were selected"
                )
            signature_callback = build_gradient_signature_callback(
                selected_layers=selected_layers,
                projection_dim=config.gradient_signature_dim,
                every_steps=config.gradient_signature_every_steps,
                start_step=config.gradient_signature_start_step,
                seed=config.seed,
                artifact_dir=config.output_dir / "gradient_signatures",
                log_metrics=tracking_run.log_metrics,
                callback_base=transformers.TrainerCallback,
            )
            trainer.add_callback(signature_callback)
        fixed_probe_callback = None
        if config.fixed_prompt_probe_sample_count > 0:
            fixed_probe_callback = build_fixed_prompt_signature_callback(
                selected_layers=selected_layers,
                examples=probe_examples,
                steps=config.fixed_prompt_probe_steps,
                projection_dim=config.gradient_signature_dim,
                seed=config.seed,
                artifact_dir=config.output_dir / "fixed_prompt_signatures",
                data_collator=collator,
                torch_module=torch,
                callback_base=transformers.TrainerCallback,
            )
            trainer.add_callback(fixed_probe_callback)
        train_output = trainer.train()
        tracking_run.log_metrics(getattr(train_output, "metrics", {}))
        if signature_callback is not None:
            plan.metadata["gradient_signatures"] = signature_callback.finalize()
        if fixed_probe_callback is not None:
            plan.metadata["fixed_prompt_signatures"] = fixed_probe_callback.finalize()
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


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"
