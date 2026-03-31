from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """FINIAS system configuration. All values from environment variables."""

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "finias"
    postgres_user: str = "finias"
    postgres_password: str = ""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # API Keys
    anthropic_api_key: str = ""
    polygon_api_key: str = ""
    fred_api_key: str = ""

    # Claude Configuration
    claude_model: str = "claude-sonnet-4-20250514"
    claude_model_fast: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 4096

    # System
    log_level: str = "INFO"
    environment: str = "development"  # development | staging | production

    @property
    def postgres_dsn(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    model_config = {"env_prefix": "FINIAS_", "env_file": ".env", "extra": "ignore"}


# Singleton
_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
