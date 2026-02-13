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

---

## E2E Tests (End-to-End)

### Overview

E2E tests run against a **live ZoneMinder instance**. They catch bugs that unit tests miss because unit tests mock HTTP responses with hand-crafted JSON fixtures.

| Category | Example | Why unit tests miss it |
|----------|---------|----------------------|
| Response structure drift | ZM returns `StartDateTime` but pyzm expects `StartTime` | Fixture uses whatever pyzm expects |
| flash() instead of JSON | `MonitorsController.delete()` returns HTML redirect | Fixture mocks clean JSON |
| Filter URL building bugs | pyzm builds wrong filter path -> wrong results | Mock returns whatever you tell it |
| Type coercion mismatches | ZM returns `"1"` (string), pyzm assumes `int` | Fixture can use any type |
| Auth flow against real server | JWT format, token refresh timing, credential format | Mock always returns 200 |

### E2E Environment Setup

| Variable | Required | Description |
|----------|----------|-------------|
| `ZM_API_URL` | Yes | Full API URL, e.g. `https://zm.local/zm/api` |
| `ZM_USER` | Yes | ZoneMinder username |
| `ZM_PASSWORD` | Yes | ZoneMinder password |
| `ZM_E2E_WRITE` | No | Set to `1` to enable write-tier tests |

If env vars are unset, all E2E tests are skipped automatically.

### Running E2E Tests

```bash
# Readonly only (safe, no data changes)
ZM_API_URL=https://zm.local/zm/api ZM_USER=admin ZM_PASSWORD=secret \
  pytest tests/e2e/ -m e2e_readonly

# Write tests (creates/modifies/deletes with cleanup)
ZM_API_URL=https://zm.local/zm/api ZM_USER=admin ZM_PASSWORD=secret ZM_E2E_WRITE=1 \
  pytest tests/e2e/ -m e2e_write

# All E2E
ZM_API_URL=https://zm.local/zm/api ZM_USER=admin ZM_PASSWORD=secret ZM_E2E_WRITE=1 \
  pytest tests/e2e/

# Collect-only (verify discovery without a live instance)
pytest tests/e2e/ -v --co -m e2e_readonly
```

### E2E Tiers

- **Readonly** (`e2e_readonly`): list, find, get, filter operations. Safe to run repeatedly.
- **Write** (`e2e_write`): create, modify, delete operations. Require `ZM_E2E_WRITE=1`. All write tests clean up:
  - Monitors prefixed `pyzm_e2e_test_` are deleted in teardown
  - Config values saved before mutation and restored in teardown
  - States recorded and restored after switching

### Known pyzm Bugs Documented as E2E Tests

- `Configs.find(name="nonexistent")` raises `TypeError` at `Configs.py:64` (no null check on `match`)
- Monitor/event delete may return `None` (flash redirect) instead of JSON
