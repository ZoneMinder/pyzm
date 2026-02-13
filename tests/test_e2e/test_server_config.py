"""E2E: ServerConfig validation."""

from __future__ import annotations

import pytest


class TestServerConfigValidation:

    def test_all_alone_valid(self):
        from pyzm.models.config import ServerConfig
        cfg = ServerConfig(models=["all"])
        assert cfg.models == ["all"]

    def test_all_mixed_raises(self):
        from pyzm.models.config import ServerConfig
        with pytest.raises(Exception):
            ServerConfig(models=["all", "yolov4"])

    def test_normal_models_valid(self):
        from pyzm.models.config import ServerConfig
        cfg = ServerConfig(models=["yolov4", "yolov7"])
        assert cfg.models == ["yolov4", "yolov7"]
