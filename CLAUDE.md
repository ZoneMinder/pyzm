* `pyzm` is the Python library for ZoneMinder (API, ML pipeline, logging)
* To run tests: `pytest tests/`
* If you need to access DB, configs etc, access it as `sudo -u www-data`
* Follow DRY principles for coding
* Always write simple code
* Use conventional commit format for all commits:
  * `feat:` new features
  * `fix:` bug fixes
  * `refactor:` code restructuring without behavior change
  * `docs:` documentation only
  * `chore:` maintenance, config, tooling
  * `test:` adding or updating tests
  * Scope is optional: `feat(install):`, `refactor(config):`, etc.
* If you are fixing bugs or creating new features, the process MUST be:
    - Create a GH issue (label it) - ALWAYS create it in pliablepixels/pyzm NEVER Zoneminder/pyzm
    - If developing a feature, create a branch
    - Commit changes referring the issue
    - Wait for the user to confirm before you close the issue
