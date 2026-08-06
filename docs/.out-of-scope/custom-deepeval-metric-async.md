# Custom Deepeval Metric `a_measure`

This project does not treat `a_measure` on custom deepeval metrics as speculative generality or dead code.

## Why this is in scope / not speculative

Deepeval's `BaseMetric` pattern expects custom metrics to implement **both** `measure` (sync) and `a_measure` (async). All official examples include both methods:

```python
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CustomMetric(BaseMetric):
    def measure(self, test_case: LLMTestCase) -> float:
        # sync scoring logic
        ...

    async def a_measure(self, test_case: LLMTestCase) -> float:
        # async counterpart — delegates to same scoring logic
        ...
```

The `a_measure` method is invoked when `assert_test` is called with `run_async=True`. Since deepeval may call `a_measure` depending on configuration, the method is required for interface compliance — not speculative generality.

The current implementation in `RecallAtKMetric` delegates synchronously:

```python
async def a_measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
    return self.measure(test_case, *args, **kwargs)
```

This is the correct pattern for a metric that has no actual async work (no IO-bound operations in the scoring logic).

## Prior requests

- #92 — "Speculative Generality — RecallAtKMetric.a_measure (Code Review #57, Item 44)"
