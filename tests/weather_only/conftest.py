import pytest

from common.weather_only import models, service


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    """Train all three models once into a temp dir (about 30 s)."""
    path = tmp_path_factory.mktemp("weather_only_models")
    for target in ("solar", "wind", "demand"):
        models.train_target(target, model_dir=path)
    return str(path)


@pytest.fixture()
def use_trained(trained_dir, monkeypatch):
    monkeypatch.setattr(service, "MODEL_DIR", trained_dir)
    models.clear_model_cache()
    yield trained_dir
    models.clear_model_cache()
