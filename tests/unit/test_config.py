from __future__ import annotations

import pytest

from src.infrastructure.config import AppConfig, load_config


def test_default_config():
    config = load_config()
    assert config.camera.index == 0
    assert config.pipeline.window_size_ms == 1000
    assert config.show_overlay is True


def test_load_from_yaml(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("""
camera:
  index: 2
pipeline:
  window_size_ms: 2000
  stride_ms: 1000
show_overlay: false
""")
    config = load_config(yaml_file)
    assert config.camera.index == 2
    assert config.pipeline.window_size_ms == 2000
    assert config.pipeline.stride_ms == 1000
    assert config.show_overlay is False


def test_partial_yaml_uses_defaults(tmp_path):
    yaml_file = tmp_path / "partial.yaml"
    yaml_file.write_text("camera:\n  index: 1\n")
    config = load_config(yaml_file)
    assert config.camera.index == 1
    assert config.pipeline.window_size_ms == 1000
    assert config.show_overlay is True


def test_empty_yaml_uses_defaults(tmp_path):
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")
    config = load_config(yaml_file)
    assert config.camera.index == 0


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")
