"""Application configuration loaded from environment / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telegram
    telegram_bot_token: str
    # Restrict the bot to these chat ids (comma-separated in env). Empty = allow all.
    allowed_chat_ids: list[int] = []

    # Anthropic
    anthropic_api_key: str
    router_model: str = "claude-haiku-4-5"
    answer_model: str = "claude-sonnet-4-6"

    # Storage
    data_dir: Path = PROJECT_ROOT / "data"
    media_dir: Path = PROJECT_ROOT / "media"
    # Default location of the Telegram Desktop export (M3 backfill)
    export_dir: Path = PROJECT_ROOT / "export"
    # LLM Wiki (M5): Layer 2 (rebuildable, gitignored) and Layer 3 (human-edited)
    wiki_dir: Path = PROJECT_ROOT / "wiki"
    wiki_guide_path: Path = PROJECT_ROOT / "schema" / "wiki_guide.md"

    # Query engine
    max_agent_iterations: int = 6
    recent_window_default_hours: int = 48
    # Don't answer @mentions older than this (backlog after laptop sleep) —
    # they are still archived. 0 = answer everything regardless of age.
    answer_max_age_minutes: int = 30

    # Transcription (M2) — local faster-whisper, CPU-only on Apple Silicon
    whisper_model: str = "large-v3-turbo"  # fallback option: "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_lang: str = "ru"
    models_dir: Path = PROJECT_ROOT / "models"
    job_max_attempts: int = 3

    # Image captioning (M4) — Haiku vision; Batch API for the backfill
    caption_model: str = "claude-haiku-4-5"
    caption_max_tokens: int = 256
    caption_batch_chunk: int = 500  # requests per batch (256MB API cap headroom)
    caption_poll_seconds: int = 30
    caption_resize_long_edge: int = 1024  # resize before sending (token cost ↓)
    caption_jpeg_quality: int = 80

    # Nightly digest / wiki maintenance (M5) — Haiku, never Opus
    digest_model: str = "claude-haiku-4-5"
    digest_max_tokens: int = 2048
    max_digest_iterations: int = 8
    # Which chat the digest summarises. Empty → first of allowed_chat_ids.
    digest_chat_id: int | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chat.db"


def get_settings() -> Settings:
    return Settings()
