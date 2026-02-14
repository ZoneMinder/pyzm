"""E2E: StreamConfig.from_dict with various YAML-style inputs."""

from __future__ import annotations


class TestStreamConfigFromDict:

    def test_default_values(self):
        from pyzm.models.config import StreamConfig
        sc = StreamConfig()
        assert sc.frame_set == ["snapshot", "alarm", "1"]
        assert sc.resize == 800
        assert sc.max_frames == 0

    def test_resize_no(self):
        from pyzm.models.config import StreamConfig
        sc = StreamConfig.from_dict({"resize": "no"})
        assert sc.resize is None

    def test_resize_number(self):
        from pyzm.models.config import StreamConfig
        sc = StreamConfig.from_dict({"resize": "640"})
        assert sc.resize == 640

    def test_frame_set_string(self):
        from pyzm.models.config import StreamConfig
        sc = StreamConfig.from_dict({"frame_set": "snapshot,alarm,1,2,3"})
        assert sc.frame_set == ["snapshot", "alarm", "1", "2", "3"]

    def test_frame_set_list(self):
        from pyzm.models.config import StreamConfig
        sc = StreamConfig.from_dict({"frame_set": ["snapshot", 5, 10]})
        assert sc.frame_set == ["snapshot", "5", "10"]

    def test_bool_yes_no(self):
        from pyzm.models.config import StreamConfig
        sc = StreamConfig.from_dict({
            "download": "yes",
            "save_frames": "no",
            "disable_ssl_cert_check": "true",
        })
        assert sc.download is True
        assert sc.save_frames is False
        assert sc.disable_ssl_cert_check is True

    def test_int_fields(self):
        from pyzm.models.config import StreamConfig
        sc = StreamConfig.from_dict({
            "max_frames": "5",
            "start_frame": "2",
            "frame_skip": "3",
            "delay": "10",
        })
        assert sc.max_frames == 5
        assert sc.start_frame == 2
        assert sc.frame_skip == 3
        assert sc.delay == 10

    def test_ignores_unknown_keys(self):
        from pyzm.models.config import StreamConfig
        sc = StreamConfig.from_dict({
            "api": "http://example.com",
            "polygons": [],
            "mid": "1",
            "frame_strategy": "most",
            "resize": "800",
        })
        assert sc.resize == 800
