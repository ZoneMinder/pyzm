# pyzm Testing Strategy

## Philosophy

pyzm testing follows a **public API first** approach:

- **Public API tests** (`tests/unit/test_api_*.py`) test `ZMApi` methods as consumers use them. All HTTP is mocked via `responses`. These are the **primary coverage layer** — since `zm_api.monitors()` creates `Monitors` → `Monitor` objects from API data, the happy path for helper accessors is already covered here.
- **Helper edge-case tests** (`tests/unit/helpers/`) cover ONLY edge cases, pure logic, and error paths not reachable through the public API. Examples: `States.find()` case-insensitive search, `Monitor.set_parameter()` payload construction.
- **Integration tests** (`tests/integration/`) chain multiple API calls in realistic workflows.

**Rule: if an API test already asserts a helper behavior, don't write a separate helper test for it.**

## Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── fixtures/
│   └── responses/              # JSON response fixtures mirroring ZM API payloads
│       ├── login_success.json
│       ├── login_legacy.json
│       ├── monitors.json
│       ├── events.json
│       ├── states.json
│       ├── configs.json
│       ├── version.json
│       └── daemon_status.json
├── unit/
│   ├── test_api_auth.py        # Login flows, token refresh, relogin
│   ├── test_api_methods.py     # monitors(), events(), states(), configs(), etc.
│   ├── test_api_request.py     # _make_request retry, error handling, content types
│   └── helpers/
│       ├── test_monitor.py     # set_parameter payload, arm/disarm URLs
│       ├── test_events.py      # URL filter building, pagination
│       ├── test_state.py       # active() false, definition() None
│       ├── test_states.py      # find() search logic
│       └── test_base.py        # ConsoleLog level filtering, exit calls
└── integration/
    └── test_api_workflow.py    # Full login -> monitors -> events -> states
```

## Fixture Patterns

### JSON Response Fixtures

Response fixtures in `tests/fixtures/responses/` mirror actual ZM API payloads. They are loaded by helper functions in `conftest.py` and provided as pytest fixtures.

### Key Shared Fixtures

| Fixture | Purpose |
|---|---|
| `zm_options` | Standard config dict for JWT auth |
| `zm_options_no_auth` | Config without credentials |
| `zm_api` | Pre-authenticated `ZMApi` with JWT login mocked |
| `zm_api_legacy` | Pre-authenticated `ZMApi` with legacy credentials |
| `suppress_logger` | (autouse) Patches `g.logger` to silent mock |
| `no_exit` | (autouse) Patches `builtins.exit` |

### Test Isolation

Each test that makes HTTP calls uses either:
- `@responses.activate` decorator for full control
- The `zm_api` fixture which uses `responses.RequestsMock` context manager

The `suppress_logger` and `no_exit` fixtures are `autouse=True` — they apply to all tests automatically.

## Mocking Strategy

### HTTP Mocking with `responses`

We use the `responses` library (not `requests-mock` or VCR):
- Simple decorator/context manager API
- No cassettes to maintain
- Best ergonomics for `requests`-based code

### Why Not VCR?

VCR records real HTTP interactions. We don't have a live ZM server in CI, and maintaining cassettes adds complexity without value here.

### Global State

pyzm uses a global logger at `pyzm.helpers.globals.logger`. The `suppress_logger` autouse fixture replaces it with a `MagicMock` for every test, preventing console spam and avoiding test interdependence.

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=pyzm --cov-report=term-missing

# Only unit tests
pytest tests/unit/ -v

# Only integration tests
pytest tests/integration/ -v -m integration

# Specific test file
pytest tests/unit/test_api_auth.py -v
```

## Coverage Targets

- Core modules (`api.py`, helpers): 80%+
- Excluded from coverage: `pyzm/ml/*`, `ZMLog.py`, `ZMMemory.py`, `ZMEventNotification.py`, `helpers/Media.py`, `helpers/utils.py`

## Testing Challenges Specific to pyzm

| Challenge | Solution |
|---|---|
| `ZMApi.__init__` calls `_login()` | Every test creating a `ZMApi` must mock the login endpoint via `responses` |
| `g.logger` global mutable state | `conftest.py` autouse fixture patches it to a silent mock |
| `ConsoleLog.Fatal()`/`Panic()` call `exit()` | Autouse fixture patches `builtins.exit` |
| `options={}` mutable default args | Tests always pass `.copy()` of options dicts |
| `Event.py` imports `progressbar` | Module-level import; `progressbar2` in test deps |
| `utils.py` imports `cv2`/`numpy` | Excluded from coverage; tested separately if needed |

## What We Explicitly Do NOT Test (Yet)

- `pyzm.ml.*` — ML modules with heavy deps (cv2, TensorFlow, dlib)
- `pyzm.ZMLog` — Requires MySQL connection
- `pyzm.ZMMemory` — Requires shared memory segments
- `pyzm.ZMEventNotification` — Requires WebSocket
- `pyzm.helpers.Media` — Requires cv2/numpy
- `pyzm.helpers.utils` — `draw_bbox` requires cv2; `Timer`/`read_config`/`template_fill` are simple utilities
