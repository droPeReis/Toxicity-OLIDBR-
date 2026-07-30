from src.api.settings import Settings


def test_settings():
    settings = Settings()
    assert type(settings) == Settings
