from src.ml.environments import EnvironmentVariables


def test_environment():
    env = EnvironmentVariables()
    assert type(env) == EnvironmentVariables
