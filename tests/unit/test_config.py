from __future__ import annotations

import tempfile
from pathlib import Path

from src.infrastructure.config import AppConfig, load_config


def test_default_config():
    config = load_config()
    assert config.camera.index == 0
    assert config.pipeline.window_size_ms == 1000
    assert config.show_overlay is True


def test_load_from_yaml():
    yaml_content = """
camera:
  index: 2
pipeline:
  window_size_ms: 2000
  stride_ms: 1000
show_overlay: false
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_config(f.name)

    assert config.camera.index == 2
    assert config.pipeline.window_size_ms == 2000
    assert config.pipeline.stride_ms == 1000
    assert config.show_overlay is False


def test_partial_yaml_uses_defaults():
    yaml_content = """
camera:
  index: 1
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_config(f.name)

    assert config.camera.index == 1
    assert config.pipeline.window_size_ms == 1000  # default
    assert config.show_overlay is True  # default
