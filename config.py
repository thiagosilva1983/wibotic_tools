from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "")
    sync_interval_seconds: int = int(os.getenv("SYNC_INTERVAL_SECONDS", "120"))
    ui_refresh_seconds: int = int(os.getenv("UI_REFRESH_SECONDS", "60"))
    app_title: str = os.getenv("APP_TITLE", "Wibotic Shared Weekly Production")

    sos_base_url: str = os.getenv("SOS_BASE_URL", "")
    sos_client_id: str = os.getenv("SOS_CLIENT_ID", "")
    sos_client_secret: str = os.getenv("SOS_CLIENT_SECRET", "")
    sos_refresh_token: str = os.getenv("SOS_REFRESH_TOKEN", "")
    sos_use_mock: bool = os.getenv("SOS_USE_MOCK", "true").lower() == "true"


settings = Settings()
