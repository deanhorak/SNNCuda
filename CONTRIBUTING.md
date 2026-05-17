# Contributing

SNNCuda is early-stage. Contributions should keep the codebase simple, measurable, and easy to reason about.

## Local Workflow

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Expectations

- Keep changes scoped to one clear purpose.
- Add tests for behavior changes.
- Prefer deterministic examples and benchmark harnesses.
- Do not tune downstream decision logic without measuring upstream signal quality.
- Document significant design choices in `docs/DECISIONS.md`.

## Style

- C++20
- Public headers under `include/snncuda`
- Implementations under `src`
- `snake_case` for functions and variables
- `PascalCase` for types
- No global mutable state unless there is a measured reason

