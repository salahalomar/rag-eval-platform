"""Process-level environment configuration -- deliberately separate from RetrievalConfig.

The distinction matters and is worth stating plainly: `RetrievalConfig` describes an
experiment and is embedded verbatim in every evaluation result, while `Settings`
describes where the process happens to be running. A database URL does not change what
gets retrieved, so recording it alongside a metric would be noise; a chunk size does,
so it lives in the config.

ENGINEERING.md forbids the retrieval path from reading environment variables directly.
This module is the single place in the library that touches the environment at all.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-derived settings, read once per process."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that a notebook, the API and the eval runner all observe one consistent
    view of the environment rather than re-reading `.env` at arbitrary points and
    potentially disagreeing mid-run.
    """
    return Settings()
