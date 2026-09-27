# Token Budget Audit

Run the audit against saved Math Tutor generation artifacts:

```bash
uv run python -m math_tutor.token_budget_audit artifacts/<run-or-attempt-root> --format markdown
```

For JSON output:

```bash
uv run python -m math_tutor.token_budget_audit artifacts/<run-or-attempt-root> --format json --output token-budget-audit.json
```

The audit is read-only. It measures the token budget already recorded by provider responses and links failed examples to local artifact files.

## Measured Phases

- `prompt`: provider-reported prompt tokens for the direct base request or specialist draft request.
- `draft`: provider-reported specialist completion tokens from `specialist_generation.json`.
- `normalization`: provider-reported prompt tokens for the base normalizer sub-job, usually `<job-id>-normalized/generation.json`.
- `output`: provider-reported completion tokens from the final generated lesson output.

The report also includes `total_provider_tokens`, which sums every provider call found on the route, including failed normalization attempts plus successful base or silent fallbacks.

When provider usage is missing, the audit falls back to a lightweight text estimate and labels that source as `estimated:*`. Do not treat estimated counts as provider billing or model-context truth.

## Failure Links

Failure examples include:

- job id and failure stage;
- prompt, draft, normalization, output, and total provider token counts;
- local artifact paths for `generation.json`, `specialist_generation.json`, `render.json`, `raw_response.txt`, and `specialist_response.txt` when present.

These links are intended to connect truncation or context-pressure symptoms, such as `finish_reason == "length"` or near-4096 completion counts, to the exact saved output that failed.

## Recommended Limits

The command reports observed max and p95 counts per phase and recommends keeping any future caps above the successful p95 before enforcement. These recommendations are documentation only. This ticket does not change runtime prompts, context windows, `max_tokens`, routing, fallback behavior, or provider settings.
