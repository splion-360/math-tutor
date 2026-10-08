"""Define provider connections and runtime policy for the Math Tutor service.
Environment loading is limited to credentials and deployment endpoints."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class _EnvironmentSettings(BaseSettings):
    """Load provider credentials and deployment endpoints from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    elevenlabs_api_key: SecretStr | None = None
    modal_vllm_base_url: str | None = None
    modal_vllm_api_key: SecretStr | None = None
    modal_visual_model_base_url: str | None = None
    modal_visual_model_api_key: SecretStr | None = None


class Settings(BaseModel):
    """Combine environment connections with code-defined runtime policy."""

    model_config = ConfigDict(extra="ignore")

    elevenlabs_api_key: SecretStr | None = None
    modal_vllm_base_url: str | None = None
    modal_vllm_api_key: SecretStr | None = None
    modal_visual_model_base_url: str | None = None
    modal_visual_model_api_key: SecretStr | None = None
    modal_vllm_timeout_seconds: float = Field(default=120, gt=0)
    modal_specialist_timeout_seconds: float = Field(default=60, gt=0)
    modal_visual_model_timeout_seconds: float = Field(default=45, gt=0)
    elevenlabs_voice_id: str = "Xb7hH8MSUJpSbSDYk0k2"
    artifact_root: Path = Path("artifacts")
    render_timeout_seconds: float = Field(default=90, gt=0)
    max_pending_jobs: int = Field(default=8, gt=0)
    validation_max_repair_attempts: int = Field(default=1, ge=0, le=1)
    spatial_unsafe_margin: float = Field(default=0.25, ge=0)
    spatial_max_width_ratio: float = Field(default=0.9, gt=0, le=1)
    spatial_max_height_ratio: float = Field(default=0.9, gt=0, le=1)
    spatial_severe_overlap_ratio: float = Field(default=0.35, gt=0, le=1)
    spatial_persistent_checkpoints: int = Field(default=2, gt=0)

    @classmethod
    def from_environment(cls, *, env_file: str | Path | None = ".env") -> Settings:
        """Load provider connections while retaining code-defined policy.

        Args:
            env_file: Optional dotenv file containing provider connection values.

        Returns:
            Settings with environment connections and code-defined policy defaults.
        """
        connections = _EnvironmentSettings(  # type: ignore[call-arg]
            _env_file=env_file
        )
        return cls(**connections.model_dump())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-cached runtime settings."""
    return Settings.from_environment()
