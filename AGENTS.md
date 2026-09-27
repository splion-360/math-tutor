# Repository Guidance

## Development principles

All implementation work in this repository should follow these rules unless the existing repository has a stronger established convention:

- Follow SOLID, DRY, KISS, and YAGNI. Prefer the simplest design that cleanly supports the current requirements.
- Do not introduce abstractions for hypothetical future requirements.
- Keep modules cohesive and dependencies explicit. Separate domain logic, infrastructure integrations, UI concerns, and model/provider adapters where practical.
- Prefer composition over inheritance unless inheritance is clearly the simpler design.
- Avoid duplicated business logic. Shared behavior should have one clear source of truth.
- Preserve deterministic logic outside the LLM wherever possible. Validation, state transitions, metrics, checkpointing, and experiment bookkeeping should not depend on model judgment unless the design explicitly requires it.
- Keep sponsor-specific code behind narrow interfaces so that the core LevelForge logic is not coupled to a particular provider.
- Do not add a dependency unless it materially reduces implementation complexity or provides functionality we genuinely need.
- Match existing repository conventions before inventing new ones.
- Use clear type definitions and explicit interfaces for important domain objects.
- Handle expected failures explicitly. Do not silently swallow exceptions or use broad exception handling without a concrete reason.
- Keep functions small enough to have one clear responsibility, but do not fragment straightforward logic into unnecessary helper functions.
- Add tests around deterministic core behavior and important state transitions. Prefer focused unit tests and a small number of end-to-end integration tests over excessive mocking.

## Documentation and comments

Keep documentation deliberately concise.

- Every source file should begin with a two-line description of that file's current responsibility in the project.
- Public or non-trivial functions and classes should use Google-style docstrings.
- Docstrings should explain the current contract, arguments, return values, raised exceptions when relevant, and non-obvious behavior. They should not contain implementation history.
- Comments should explain why something non-obvious exists, not restate what the code is visibly doing.
- Do not add comments or docstrings describing rejected approaches, previous iterations, temporary scope decisions, or technologies that are intentionally absent unless that information is necessary to understand the current implementation.
- Do not use comments as a changelog.
- Do not write promotional language such as "robust", "production-grade", "powerful", or "scalable" unless the code or measured results specifically justify the claim.
- Avoid large explanatory block comments when clear naming and decomposition can make the code self-explanatory.

## Repository hygiene

- Do not modify unrelated files while implementing a feature.
- Do not perform broad refactors unless they are required for the current task.
- Do not rename public APIs or reorganize directories without a concrete benefit and an explanation beforehand.
- Do not commit generated artifacts, secrets, credentials, local environment files, large model assets, or temporary debugging output.
- Keep configuration separate from implementation where configuration is expected to vary.
- Use environment variables for secrets and external-service credentials.
- Add or update `.env.example` entries when new configuration is required, but never populate them with real credentials.
- Before adding new tooling, inspect the existing formatter, linter, test framework, package manager, and project structure and use what is already present whenever reasonable.

## Working style

This repository is often developed in a Herdr environment.

- You may delegate independent research or repository-inspection tasks to separate agents.
- You may use Claude or another review tab as a second programmer/reviewer when there is genuine uncertainty, an architectural tradeoff, or a useful independent review opportunity.
- Do not delegate merely to create activity. Use additional agents when parallel work materially reduces uncertainty or saves time.
- Reconcile delegated findings yourself before presenting them as conclusions.
- If you discover an ambiguity that materially changes the architecture, surface it rather than quietly assuming an answer.
- Once an initial architecture review is complete, stop and report the findings. Do not begin implementation until the architecture and unresolved assumptions have been agreed upon.
