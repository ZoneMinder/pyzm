# Changelog

All notable changes to this project will be documented in this file.


## [2.0.3] - 2026-02-14

### Bug Fixes

- add native YOLO26 end-to-end output parsing with pre-NMS fallback ([575247b](https://github.com/pliablepixels/pyzm/commit/575247b98cfd1c795f4f0fa938a51b37a998ef5f))
- letterbox preprocessing and xyxy coord format for ONNX YOLO models ([96198d5](https://github.com/pliablepixels/pyzm/commit/96198d5670d042ec6081205f4fe1415883cfe347))
- eagerly load backends in _ensure_pipeline() for serve ([445b745](https://github.com/pliablepixels/pyzm/commit/445b74507918051291e3c425db7a9a59c3196050))
- add sphinx_rtd_theme to docs requirements and _static dir ([5ac9bf2](https://github.com/pliablepixels/pyzm/commit/5ac9bf20a7e4f2eb2781b36e024affae15c0b90d))
- restore pyzm.ml.alpr and pyzm.ml.face in setup.py ([260f666](https://github.com/pliablepixels/pyzm/commit/260f66679db1e2ba96aad68ac47c9e26983d9e9b))
- handle non-UTF-8 bytes when decoding shared memory strings ([4ba0ee5](https://github.com/pliablepixels/pyzm/commit/4ba0ee502d7a2ab9644b7d869020df32d8eab5c4))
- don't skip file/console logging when DB reflection fails ([de16b5d](https://github.com/pliablepixels/pyzm/commit/de16b5daed8f34475da74fa039b3669278ce8a89))
- strip whitespace from stream parameter to handle non-breaking spaces ([3874a26](https://github.com/pliablepixels/pyzm/commit/3874a26fce03d3837062d33949810bf7edf08636))
- warn when OpenCV version is too old for ONNX YOLOv26 models ([a913280](https://github.com/pliablepixels/pyzm/commit/a913280643d0480bd5a88af5b2ae15eec35cf329))
- apply defaults for unset env vars and handle missing zm.conf gracefully ([47a8cea](https://github.com/pliablepixels/pyzm/commit/47a8ceaf506a7f08b5da5446bebaab85931329c2))
- fall back to CPU when CUDA backend is unavailable for YOLO inference ([0223dd1](https://github.com/pliablepixels/pyzm/commit/0223dd1482c3fed0a89da28804bd082fe0c8c3e9))
- prefix all log messages with model name for clarity ([84e6574](https://github.com/pliablepixels/pyzm/commit/84e6574c4d045466591e296178a8c0d457465121))
- expose lock_name on wrapper classes for dedup logic ([8db190a](https://github.com/pliablepixels/pyzm/commit/8db190a41d10c69931b95e5efa3fb2f2d59abc5a))
- prevent deadlock when multiple models share the same GPU lock ([2a84e90](https://github.com/pliablepixels/pyzm/commit/2a84e90531ac60cb2f12f52b3b3ef90758088a35))
- Later versions of OpenCV have changed format for layers ([271d111](https://github.com/pliablepixels/pyzm/commit/271d111c02b208e53b8cd06aeba60862c2508067))

### Documentation

- add developer release notes and --skip-pypi flag ([c726bce](https://github.com/pliablepixels/pyzm/commit/c726bcefdf6b4c40cbe61bc5b04375c1f3b573c7))
- expand sidebar navigation to show subsection indicators ([09c4948](https://github.com/pliablepixels/pyzm/commit/09c494816440ac0eb531dc751e862b0db5103672))
- specify OpenCV 4.13+ requirement for ONNX YOLO models ([407422d](https://github.com/pliablepixels/pyzm/commit/407422d73d429ebe962c6cd53053193caa8e9451))
- link README installation section to RTD guide ([65f67dd](https://github.com/pliablepixels/pyzm/commit/65f67dd563378b7306581aab9f8a3f128fd02c67))
- add installation guide with --break-system-packages note ([b4f15d9](https://github.com/pliablepixels/pyzm/commit/b4f15d9c13e549b388d4fa928df7af5fb8fb99f2))
- fix ES link to point to v7+ RTD ([5e095c6](https://github.com/pliablepixels/pyzm/commit/5e095c6e425f0261720b9e20f658cc38bd6aa661))
- add ES v7+ and zmNg to sidebar ([6bba6e9](https://github.com/pliablepixels/pyzm/commit/6bba6e92c447f745ec686960920a79e8fa33dd97))
- auto-update copyright year in Sphinx config ([c958458](https://github.com/pliablepixels/pyzm/commit/c95845808b1d8815ced9f2d50520074c6946451f))
- add related projects section with ES, zmNg, and zmNinja links ([6e9cdc7](https://github.com/pliablepixels/pyzm/commit/6e9cdc7f1fa65728de2e71fab831370d6d6dd87c))
- add testing guide to README and RTD ([a3706cf](https://github.com/pliablepixels/pyzm/commit/a3706cf039c0d432343c3ed4f63fc5b2bf5fc59b))
- fix EventStartCommand location in serve.rst ([1577fe2](https://github.com/pliablepixels/pyzm/commit/1577fe22bbc67a85330d8b173e76bb1aee657a4c))
- point README to pyzmv2.readthedocs.io ([87ecbf1](https://github.com/pliablepixels/pyzm/commit/87ecbf13e2ad2e76ef74b49fce6de12ea6aefbaf))
- fix ReadTheDocs build for pyzm.serve autodoc ([a6ab684](https://github.com/pliablepixels/pyzm/commit/a6ab684d3ce3b76e18252f89a383cbaa16331d40))
- clarified never to hit ZM ([6a54eae](https://github.com/pliablepixels/pyzm/commit/6a54eae62b706bc6be96321145547f8097543ce4))

### Features

- add yolo11n, yolo11s, yolo26n, yolo26s model presets ([2843619](https://github.com/pliablepixels/pyzm/commit/2843619436c2b74326c97ef150d5d207a31b9a31))
- add --models all lazy loading and overhaul serve docs ([4596f63](https://github.com/pliablepixels/pyzm/commit/4596f6339c8e489a41db867e8aed65b75c7af5a8))
- add URL-mode remote detection for pyzm.serve ([297149d](https://github.com/pliablepixels/pyzm/commit/297149dfaa618dfeb56e729cbe0b64e24e243bd6))
- add pyzm.serve remote ML detection server and Detector gateway mode ([8ee09bf](https://github.com/pliablepixels/pyzm/commit/8ee09bff22d9e862b1bc21f0f97aabfde3f06ac8))
- wire up missing config fields and past-detection filtering ([5948937](https://github.com/pliablepixels/pyzm/commit/5948937cdcb9d91dca6cc5c39aa49118972db1fd))
- add ZMClient.event_path() and fix setup.py module list ([c3e362c](https://github.com/pliablepixels/pyzm/commit/c3e362cd86192d1c1ade24cc27ed4f8a31b51359))
- pyzm v2 rewrite — Pydantic models, new ML pipeline, ZM API client ([b28a7a0](https://github.com/pliablepixels/pyzm/commit/b28a7a0c86254509829116ecab470a6c8cce200c))
- add ZM 1.38 SharedData struct support with size-based auto-detection ([7ee4e0b](https://github.com/pliablepixels/pyzm/commit/7ee4e0b55b37b7a57be7b487c8d62aa11ae9b360))
- add ONNX model support to Yolo class via OpenCV DNN ([8608afe](https://github.com/pliablepixels/pyzm/commit/8608afe49b79a3a3e86c09cfb95d7e68bd9c3032))

### Miscellaneous

- add release tooling (git-cliff + make_release.sh) and fix CLAUDE.md ([1bb0aa9](https://github.com/pliablepixels/pyzm/commit/1bb0aa96c9cb34be73349baded2ecafd5f044025))
- update package URLs, modernize build, bump to 2.0.1 ([957384a](https://github.com/pliablepixels/pyzm/commit/957384a475ad9f8b9a2d29944bff2e98a70d3e48))
- bump version to 0.4.1 ([1facaf1](https://github.com/pliablepixels/pyzm/commit/1facaf1dec2cc96facb3d967d8c067c523837d39))
- bump version to 0.4.0 ([d36120a](https://github.com/pliablepixels/pyzm/commit/d36120a4766a0675ea5f7f9a09ac6e82c9c2da06))
- CLAUDE.md first version ([7a0e5f8](https://github.com/pliablepixels/pyzm/commit/7a0e5f8f354884a59928d464d7d4982c716b3301))
- ver bump ([f16a30d](https://github.com/pliablepixels/pyzm/commit/f16a30db13c13fcdea24c787f7a3fe492d2300b8))

### Refactoring

- remove model presets, resolve all names from disk ([95e7fe2](https://github.com/pliablepixels/pyzm/commit/95e7fe237a270152dfad947714698d8a256c5bf1))
- update template_fill to use ${key} syntax instead of legacy {{key}} ([e034833](https://github.com/pliablepixels/pyzm/commit/e03483331d682de48fb2adce7fed850d44b6e796))
- switch read_config/get from ConfigParser to YAML ([43e9ef7](https://github.com/pliablepixels/pyzm/commit/43e9ef78c55ce8546bdff52744c54b725bd9481a))
- split Yolo into YoloDarknet and YoloOnnx subclasses ([a8f60ec](https://github.com/pliablepixels/pyzm/commit/a8f60ec57c28925f4b3931848847d326ae30d1a8))
- remove direct ultralytics backend, use ONNX via OpenCV DNN ([07de62b](https://github.com/pliablepixels/pyzm/commit/07de62b8e33df0c3a630ea2946cad1815d837a00))

### Testing

- add 89 e2e tests covering every objectconfig feature ([13b578d](https://github.com/pliablepixels/pyzm/commit/13b578dca4430d15db3ad7a7b0d31b5318bd2147))

### https

- //github.com/pliablepixels/zmeventnotification/issues/381 - add flag called convert_snapshot_to_fid ([c41b403](https://github.com/pliablepixels/pyzm/commit/c41b40387808463685256886df8c8b9659c07a82))
- //github.com/pliablepixels/zmeventnotification/issues/381 - add delay_between_frames ([4198681](https://github.com/pliablepixels/pyzm/commit/41986813b1fcbe72aa3269002f6c43864d81b7cc))
- //github.com/pliablepixels/zmeventnotification/issues/346 move to pycoral ([a4cf481](https://github.com/pliablepixels/pyzm/commit/a4cf48158a73338acff4522097d152c7f4f2a8c9))
<!-- generated by git-cliff -->
