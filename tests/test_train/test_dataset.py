"""Tests for pyzm.train.dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyzm.train.dataset import Annotation, QualityReport, YOLODataset


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

class TestAnnotation:
    def test_to_yolo_line(self):
        a = Annotation(class_id=2, cx=0.5, cy=0.3, w=0.2, h=0.4)
        line = a.to_yolo_line()
        assert line.startswith("2 ")
        parts = line.split()
        assert len(parts) == 5
        assert float(parts[1]) == pytest.approx(0.5)
        assert float(parts[2]) == pytest.approx(0.3)

    def test_from_yolo_line(self):
        a = Annotation.from_yolo_line("3 0.512345 0.678901 0.123456 0.234567")
        assert a.class_id == 3
        assert a.cx == pytest.approx(0.512345)
        assert a.cy == pytest.approx(0.678901)

    def test_roundtrip(self):
        orig = Annotation(class_id=1, cx=0.55, cy=0.45, w=0.3, h=0.6)
        line = orig.to_yolo_line()
        back = Annotation.from_yolo_line(line)
        assert back.class_id == orig.class_id
        assert back.cx == pytest.approx(orig.cx, abs=1e-5)
        assert back.cy == pytest.approx(orig.cy, abs=1e-5)
        assert back.w == pytest.approx(orig.w, abs=1e-5)
        assert back.h == pytest.approx(orig.h, abs=1e-5)


# ---------------------------------------------------------------------------
# YOLODataset
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset(tmp_path: Path) -> YOLODataset:
    ds = YOLODataset(project_dir=tmp_path / "project", classes=["person", "car", "package"])
    ds.init_project()
    return ds


@pytest.fixture
def sample_images(tmp_path: Path) -> list[Path]:
    """Create 10 fake image files."""
    imgs_dir = tmp_path / "source_images"
    imgs_dir.mkdir()
    paths = []
    for i in range(10):
        p = imgs_dir / f"img_{i:03d}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG-ish
        paths.append(p)
    return paths


class TestYOLODataset:
    def test_init_project_creates_dirs(self, dataset: YOLODataset):
        assert (dataset.project_dir / "images" / "all").is_dir()
        assert (dataset.project_dir / "labels" / "all").is_dir()
        assert (dataset.project_dir / "images" / "train").is_dir()
        assert (dataset.project_dir / "images" / "val").is_dir()
        assert (dataset.project_dir / "project.json").exists()

    def test_load_project(self, dataset: YOLODataset):
        loaded = YOLODataset.load(dataset.project_dir)
        assert loaded.classes == ["person", "car", "package"]

    def test_class_groups_persist(self, tmp_path: Path):
        groups = {"vehicle": ["car", "truck"], "person": ["person"]}
        ds = YOLODataset(
            project_dir=tmp_path / "grouped",
            classes=["person", "vehicle"],
            class_groups=groups,
        )
        ds.init_project()
        loaded = YOLODataset.load(ds.project_dir)
        assert loaded.class_groups == groups
        assert loaded.classes == ["person", "vehicle"]

    def test_add_image(self, dataset: YOLODataset, sample_images: list[Path]):
        anns = [Annotation(class_id=0, cx=0.5, cy=0.5, w=0.3, h=0.4)]
        dest = dataset.add_image(sample_images[0], anns)
        assert dest.exists()
        assert (dataset._labels_dir / f"{dest.stem}.txt").exists()

    def test_add_image_dedup(self, dataset: YOLODataset, sample_images: list[Path]):
        """Adding an image with the same name gets a counter suffix."""
        dataset.add_image(sample_images[0], [])
        # Create a different file with same name in a different dir
        other_dir = sample_images[0].parent.parent / "other"
        other_dir.mkdir()
        dup = other_dir / sample_images[0].name
        dup.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        dest = dataset.add_image(dup, [])
        assert "_1" in dest.stem

    def test_add_image_no_annotations(self, dataset: YOLODataset, sample_images: list[Path]):
        dest = dataset.add_image(sample_images[0], [])
        label = dataset._labels_dir / f"{dest.stem}.txt"
        assert label.read_text() == ""

    def test_staged_images(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images[:3]:
            dataset.add_image(img, [])
        assert len(dataset.staged_images()) == 3

    def test_annotations_for(self, dataset: YOLODataset, sample_images: list[Path]):
        anns = [
            Annotation(class_id=0, cx=0.5, cy=0.5, w=0.3, h=0.4),
            Annotation(class_id=1, cx=0.2, cy=0.3, w=0.1, h=0.2),
        ]
        dest = dataset.add_image(sample_images[0], anns)
        loaded = dataset.annotations_for(dest.name)
        assert len(loaded) == 2
        assert loaded[0].class_id == 0
        assert loaded[1].class_id == 1

    def test_update_annotations(self, dataset: YOLODataset, sample_images: list[Path]):
        dest = dataset.add_image(sample_images[0], [])
        assert dataset.annotations_for(dest.name) == []

        new_anns = [Annotation(class_id=2, cx=0.5, cy=0.5, w=0.2, h=0.3)]
        dataset.update_annotations(dest.name, new_anns)
        loaded = dataset.annotations_for(dest.name)
        assert len(loaded) == 1
        assert loaded[0].class_id == 2

    def test_split_basic(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])
        dataset.split(val_ratio=0.2)

        train_imgs = list(dataset._train_images.iterdir())
        val_imgs = list(dataset._val_images.iterdir())
        assert len(train_imgs) + len(val_imgs) == 10
        assert len(val_imgs) == 2  # 20% of 10

    def test_split_copies_labels(self, dataset: YOLODataset, sample_images: list[Path]):
        anns = [Annotation(class_id=1, cx=0.5, cy=0.5, w=0.3, h=0.4)]
        for img in sample_images[:5]:
            dataset.add_image(img, anns)
        dataset.split(val_ratio=0.2)

        train_labels = list(dataset._train_labels.iterdir())
        val_labels = list(dataset._val_labels.iterdir())
        assert len(train_labels) + len(val_labels) == 5

    def test_split_invalid_ratio(self, dataset: YOLODataset):
        with pytest.raises(ValueError, match="val_ratio"):
            dataset.split(val_ratio=0.0)
        with pytest.raises(ValueError, match="val_ratio"):
            dataset.split(val_ratio=1.0)

    def test_split_deterministic(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])

        dataset.split(val_ratio=0.3, seed=42)
        val1 = sorted(p.name for p in dataset._val_images.iterdir())

        dataset.split(val_ratio=0.3, seed=42)
        val2 = sorted(p.name for p in dataset._val_images.iterdir())

        assert val1 == val2

    def test_generate_yaml(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images[:3]:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])
        dataset.split()

        yaml_path = dataset.generate_yaml()
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "nc: 3" in content
        assert "person" in content
        assert "car" in content
        assert "package" in content
        assert "train: images/train" in content
        assert "val: images/val" in content


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------

class TestQualityReport:
    def test_small_dataset_warning(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images[:5]:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])

        report = dataset.quality_report()
        assert report.total_images == 5
        assert report.annotated_images == 5
        msgs = [w.message for w in report.warnings]
        assert any("Very small dataset" in m for m in msgs)

    def test_unannotated_warning(self, dataset: YOLODataset, sample_images: list[Path]):
        dataset.add_image(sample_images[0], [])
        dataset.add_image(sample_images[1], [Annotation(0, 0.5, 0.5, 0.3, 0.3)])

        report = dataset.quality_report()
        assert report.unannotated_images == 1
        msgs = [w.message for w in report.warnings]
        assert any("no annotations" in m for m in msgs)

    def test_few_images_warning(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images[:3]:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])

        report = dataset.quality_report()
        msgs = [w.message for w in report.warnings]
        assert any("Very few images for 'person'" in m for m in msgs)

    def test_per_class_counts(self, dataset: YOLODataset, sample_images: list[Path]):
        dataset.add_image(sample_images[0], [
            Annotation(0, 0.5, 0.5, 0.3, 0.3),
            Annotation(0, 0.2, 0.2, 0.1, 0.1),
        ])
        dataset.add_image(sample_images[1], [
            Annotation(1, 0.5, 0.5, 0.3, 0.3),
        ])

        report = dataset.quality_report()
        assert report.per_class["person"] == 2
        assert report.per_class["car"] == 1
        assert report.per_class["package"] == 0

    def test_empty_dataset(self, dataset: YOLODataset):
        report = dataset.quality_report()
        assert report.total_images == 0
        assert report.annotated_images == 0
        # Still warns about small dataset (0 < 20)
        msgs = [w.message for w in report.warnings]
        assert any("Very small dataset" in m for m in msgs)
