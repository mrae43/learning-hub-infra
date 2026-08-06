# Pytest Module-Level Test Data

This project does not treat module-level YAML/JSON I/O in test files as a defect, provided the side effect is confined to that test file.

## Why this is accepted

Pytest's `@pytest.mark.parametrize` requires its values at decoration (import) time — there is no native way to defer parametrize arguments to fixture resolution. Loading test data at module level is the standard, idiomatic pattern:

```python
_EVAL_DATA = _load_eval_data()           # YAML I/O at import time
_EVAL_QUERIES = [EvalQuery(...) for ...] # validated at import time

@pytest.mark.parametrize("query_data", _EVAL_QUERIES)
def test_something(query_data):
    ...
```

Fixing this "cosmetic concern" would require one of:

- **Isolating eval tests in `tests/eval/`** — directory restructure for one file's import order
- **`pytest-lazy-fixtures` dependency** — new third-party dep to defer parametrize resolution
- **`pytest_generate_tests` hook** — metafunc-based parametrization, significantly more complex

None of these add meaningful value. The side effect is confined to one test file, runs only during test collection, and causes no runtime failures.

## Prior requests

- #93 — "Module-level side effects — test_eval.py YAML I/O at import time (Code Review #57, Item 45)"
