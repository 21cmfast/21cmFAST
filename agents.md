# AGENTS.md

## Do
- use uv run for all project tooling commands (pytest, ruff, etc.)
- setup environment with `CC=gcc uv pip install -e ".[dev]"`
- add type hints for all new parameters
- write or update tests for every behavior change
- add numpydoc-style docstrings for new public modules, classes, and functions
- update docstrings whenever parameters are added or changed
- prefer minimal, file-scoped checks before broad checks

## Do not
- do not edit CHANGELOG.rst unless explicitly asked
- do not modify regression artifacts or data snapshots unless explicitly asked
- do not run full test suites or expensive notebook/doc executions without approval
- do not introduce API-breaking renames or default changes without explicit confirmation
- do not ignore ruff errors by using `# noqa` on those lines.
- do not modify the ruff linting rules in pyproject.toml without explicit confirmation

### Commands

#### File-scoped quality checks (preferred)
uv run ruff format path/to/file.py
uv run ruff check --fix path/to/file.py

#### Targeted tests (preferred)
uv run pytest tests/path/to/test_file.py
uv run pytest tests/path/to/test_file.py -k keyword

#### Wider checks (ask first)
uv run prek run -a
uv run pytest

### Safety and permissions

Allowed without prompt:
- read, list, and search files
- edit source, tests, and docs text files relevant to the task
- run file-scoped ruff checks
- run one or a few targeted pytest files

Ask first:
- package or environment installs/upgrades
- full-suite pytest runs
- notebook execution across docs/examples
- regenerating regression data
- deleting files, chmod, git push

### Testing expectations

- bug fixes should include regression tests
- use explicit numeric tolerances for floating-point comparisons
- handle warnings intentionally (fix or filter in tests)

### API docs

- docs are built with sphinx and numpydoc
- keep API docs indexed in docs/api.rst when adding new public objects

### Process note

- branch/release docs and workflows may differ historically; do not assume release branching strategy unless user specifies

### PR checklist

- formatting and lint checks pass
- unit tests pass
- tests added for new or changed behavior

### When stuck

- ask a clarifying question
- propose a short plan with assumptions
- provide a minimal draft patch with explicit unknowns

### Test-first mode

- for new features, write or update tests first, then code to green

## Tips for fast debugging

- Make use of the cli where possible, e.g. `21cmfast run lightcone` (docs at docs/tutorials/cli_usage.rst)
- Running a full lightcone or coeval simulation takes a lot of time and resources. Consider the following as applicable:

- If possible run a coeval box at a high redshift to limit the redshift evolution required, e.g. `21cmfast run coeval --redshift 12.0`
- Make `HII_DIM` small and set `LOWRES_CELL_SIZE_MPC` a bit higher
- Make `HIRES_TO_LOWRES_FACTOR` small (like 1 or 2).
- Set `ZPRIME_STEP_FACTOR` to be large, e.g 1.1 or 1.2 even.
- Unless required by the debug task at hand, set `USE_TS_FLUCT=False`, `MASS_DEPENDENT_ZETA=False` and `SOURCE_MODEL=E-INTEGRAL`.
- To achieve many of those settings in one argument, use the "tiny" template, e.g. `21cmfast run coeval --template latest tiny`
