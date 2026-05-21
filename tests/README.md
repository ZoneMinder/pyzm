# pyzm Testing Strategy

## Core Design Intent

**The real ZoneMinder API is the source of truth.** Hand-crafted JSON fixtures can diverge from actual server responses in type coercion, field names, response structure, and error formats. E2E tests against a live ZoneMinder instance are the authoritative validation layer.

The test suite is structured in four tiers, each with a specific role:

1. **E2E tests** (`tests/e2e/`) — the gold standard. Run against a **live ZoneMinder instance** with real HTTP, real auth, real data. Every assertion is validated against actual server behavior.
2. **Unit API tests** (`tests/unit/test_api_*.py`) — the primary offline coverage layer. Test `ZMApi` public methods as consumers use them. HTTP is mocked via `responses`. Since `zm_api.monitors()` creates `Monitors` → `Monitor` objects from fixture data, the happy path for helper accessors is already covered here.
3. **Helper edge-case tests** (`tests/unit/helpers/`) — cover ONLY edge cases, pure logic, and error paths not reachable through the public API. Examples: `States.find()` case-insensitive search, `Monitor.set_parameter()` payload construction, `State.definition()` returning None for empty strings.
4. **Integration tests** (`tests/integration/`) — chain multiple mocked API calls in realistic workflows.

### Rules

- **If an API test already asserts a helper behavior, don't write a separate helper test for it.** Duplicate assertions across tiers create maintenance burden without adding confidence.
- **E2E tests intentionally re-validate behaviors covered by unit tests.** This is by design — E2E catches real-world divergence that fixtures miss. Overlap between E2E and unit tiers is expected and valuable.
- **JSON fixtures must mirror real ZM API payload structure**, including using strings for numeric fields (ZM returns `"1"` not `1` for many fields). E2E tests verify this assumption holds.

## Why E2E Tests Exist

Unit tests mock HTTP responses with hand-crafted JSON. This means unit tests **cannot catch** an entire class of bugs:

| Category | Example | Why unit tests miss it |
|----------|---------|----------------------|
| Response structure drift | ZM returns `StartDateTime` but pyzm expects `StartTime` | Fixture uses whatever pyzm expects |
| flash() instead of JSON | `MonitorsController.delete()` returns HTML redirect | Fixture mocks clean JSON |
| Filter URL building bugs | pyzm builds wrong filter path → wrong results | Mock returns whatever you tell it |
| Type coercion mismatches | ZM returns `int` but fixture uses `str` (or vice versa) | Fixture can use any type |
| Auth flow against real server | JWT format, token refresh timing, credential format | Mock always returns 200 |
| Real-world data volume | Pagination with thousands of events | Mock returns static fixture |

E2E tests are the only way to confirm pyzm works against the software it claims to wrap.

## Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures (autouse logger/exit patches)
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
│       ├── test_states.py      # find() search logic (case-insensitive, by id)
│       └── test_base.py        # ConsoleLog level filtering, exit calls
├── integration/
│   └── test_api_workflow.py    # Full login → monitors → events → states
└── e2e/
    ├── conftest.py             # Live ZM fixtures, cleanup factories
    ├── test_e2e_auth.py        # Login, version, timezone, get_auth, bad creds
    ├── test_e2e_monitors.py    # List/find/accessors, add/modify/delete
    ├── test_e2e_events.py      # List/filter/accessors, URLs, delete
    ├── test_e2e_states.py      # List/find/active invariant, set state
    ├── test_e2e_configs.py     # List/find, set with restore
    └── test_e2e_edge_cases.py  # Pagination coherence, type coercion, arm/disarm
```

## Fixture Patterns

### JSON Response Fixtures

Response fixtures in `tests/fixtures/responses/` mirror actual ZM API payloads. All numeric values are strings (e.g. `"Id": "1"`, `"Enabled": "1"`, `"Width": "1920"`) because that is what ZoneMinder's API returns. This keeps fixtures honest — if pyzm's type coercion breaks, the unit tests catch it.

### Key Shared Fixtures

| Fixture | Scope | Purpose |
|---|---|---|
| `zm_options` | function | Standard config dict for JWT auth |
| `zm_options_no_auth` | function | Config without credentials |
| `zm_api` | function | Pre-authenticated `ZMApi` with JWT login mocked |
| `zm_api_legacy` | function | Pre-authenticated `ZMApi` with legacy credentials |
| `suppress_logger` | function (autouse) | Patches `g.logger` to silent mock |
| `no_exit` | function (autouse) | Patches `builtins.exit` |

### E2E Fixtures

| Fixture | Scope | Purpose |
|---|---|---|
| `zm_options_live` | session | Options dict from env vars (skips if unset) |
| `zm_api_live` | session | Single authenticated `ZMApi` for all E2E tests |
| `zm_api_fresh` | function | Fresh login per test (auth-specific tests) |
| `e2e_monitor_factory` | function | Creates monitors with auto-cleanup in teardown |
| `e2e_config_restorer` | function | Saves config value, restores in teardown |
| `requires_write` | function | Skips if `ZM_E2E_WRITE != "1"` |

The E2E conftest overrides `suppress_logger` and `no_exit` with no-ops so E2E tests use the real logger and real `exit()`.

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
# All unit + integration tests (no live ZM needed)
pytest tests/unit/ tests/integration/ -v

# With coverage
pytest tests/ -v --cov=pyzm --cov-report=term-missing

# Only unit tests
pytest tests/unit/ -v

# Only integration tests
pytest tests/integration/ -v -m integration

# Specific test file
pytest tests/unit/test_api_auth.py -v

# E2E readonly only (safe, no data changes)
ZM_API_URL=https://zm.local/zm/api ZM_USER=admin ZM_PASSWORD=secret \
  pytest tests/e2e/ -m e2e_readonly

# E2E write tests (creates/modifies/deletes with cleanup)
ZM_API_URL=https://zm.local/zm/api ZM_USER=admin ZM_PASSWORD=secret ZM_E2E_WRITE=1 \
  pytest tests/e2e/ -m e2e_write

# All E2E
ZM_API_URL=https://zm.local/zm/api ZM_USER=admin ZM_PASSWORD=secret ZM_E2E_WRITE=1 \
  pytest tests/e2e/

# Collect-only (verify discovery without a live instance)
pytest tests/e2e/ -v --co -m e2e_readonly
```

## E2E Environment Setup

| Variable | Required | Description |
|----------|----------|-------------|
| `ZM_API_URL` | Yes | Full API URL, e.g. `https://zm.local/zm/api` |
| `ZM_USER` | Yes | ZoneMinder username |
| `ZM_PASSWORD` | Yes | ZoneMinder password |
| `ZM_E2E_WRITE` | No | Set to `1` to enable write-tier tests |

If env vars are unset, all E2E tests are skipped automatically.

### E2E Tiers

- **Readonly** (`e2e_readonly`): list, find, get, filter operations. Safe to run repeatedly.
- **Write** (`e2e_write`): create, modify, delete operations. Require `ZM_E2E_WRITE=1`. All write tests clean up:
  - Monitors prefixed `pyzm_e2e_test_` are deleted in teardown
  - Config values saved before mutation and restored in teardown
  - States recorded and restored after switching

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

## Known pyzm Bugs Documented as E2E Tests

- `Configs.find(name="nonexistent")` raises `TypeError` at `Configs.py:64` (no null check on `match`)
- Monitor/event delete may return `None` (flash redirect) instead of JSON
