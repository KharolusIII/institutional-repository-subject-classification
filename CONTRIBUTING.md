# Contributing

Contributions that improve reproducibility, evaluation, documentation, or
support for other institutional repositories are welcome.

1. Create a focused branch.
2. Install the development environment with `python -m pip install -e .[dev]`.
3. Run `pytest` before submitting a pull request.
4. Do not commit real metadata, full text, handles, predictions, caches,
   checkpoints, credentials, or local paths.
5. Add or update tests for behavioral changes.

The synthetic smoke profile must remain runnable without private data. Changes
to model selection must preserve the calibration/validation/test separation
and document any effect on the frozen experimental protocol.
