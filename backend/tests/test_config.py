"""Configuration loading.

backend/.env is the committed record of which models produced the results, so
it has to actually reach the running service — these tests pin that.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ENV_FILE, Settings, get_settings, load_env_file  # noqa: E402


def test_env_file_is_present() -> None:
    """Committed on purpose — it holds no secrets and records the model setup."""
    assert ENV_FILE.is_file(), f"expected {ENV_FILE} to be committed"


def test_env_file_declares_the_models() -> None:
    text = ENV_FILE.read_text(encoding="utf-8")
    assert "LSD_VISION_MODEL=" in text
    assert "LSD_TEXT_MODEL=" in text


def test_settings_pick_up_the_env_file() -> None:
    settings = get_settings()
    declared = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            declared[key.strip()] = value.strip()
    assert settings.vision_model == declared["LSD_VISION_MODEL"]
    assert settings.text_model == declared["LSD_TEXT_MODEL"]


def test_real_environment_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LSD_TEST_KEY=from_file\n", encoding="utf-8")

    monkeypatch.setenv("LSD_TEST_KEY", "from_environment")
    load_env_file(env_file)
    assert os.environ["LSD_TEST_KEY"] == "from_environment"

    monkeypatch.delenv("LSD_TEST_KEY", raising=False)
    load_env_file(env_file)
    assert os.environ["LSD_TEST_KEY"] == "from_file"
    monkeypatch.delenv("LSD_TEST_KEY", raising=False)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("LSD_TEST_PARSE=plain", "plain"),
        ('LSD_TEST_PARSE="double quoted"', "double quoted"),
        ("LSD_TEST_PARSE='single quoted'", "single quoted"),
        ("LSD_TEST_PARSE=with_value # trailing comment", "with_value"),
        ("export LSD_TEST_PARSE=exported", "exported"),
        ("  LSD_TEST_PARSE=indented  ", "indented"),
        # A '#' with no preceding space is part of the value, not a comment.
        ("LSD_TEST_PARSE=tag#1", "tag#1"),
        # Values legitimately containing '=' (e.g. URLs with query strings).
        ("LSD_TEST_PARSE=a=b", "a=b"),
    ],
)
def test_env_line_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str, expected: str
) -> None:
    monkeypatch.delenv("LSD_TEST_PARSE", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{line}\n", encoding="utf-8")
    load_env_file(env_file)
    assert os.environ["LSD_TEST_PARSE"] == expected
    monkeypatch.delenv("LSD_TEST_PARSE", raising=False)


def test_comments_and_blank_lines_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LSD_TEST_REAL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n\n   \n#LSD_TEST_COMMENTED=nope\nLSD_TEST_REAL=yes\n",
        encoding="utf-8",
    )
    assert load_env_file(env_file) == 1
    assert os.environ["LSD_TEST_REAL"] == "yes"
    assert "LSD_TEST_COMMENTED" not in os.environ
    monkeypatch.delenv("LSD_TEST_REAL", raising=False)


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "absent.env") == 0


def test_committed_env_parses_without_surprises() -> None:
    """Guard against a stray quote or comment mangling a real value."""
    settings = Settings()
    assert settings.decision_threshold == pytest.approx(0.5)
    assert settings.llm_base_url.startswith("http://")
    assert not settings.llm_base_url.endswith("/")
    assert settings.llm_keep_alive
    assert settings.max_tokens > 0
    assert settings.llm_context_tokens >= settings.max_tokens
    assert 0.0 <= settings.llm_temperature <= 1.0
    # Every model call must stay on this machine.
    assert "127.0.0.1" in settings.llm_base_url or "localhost" in settings.llm_base_url
