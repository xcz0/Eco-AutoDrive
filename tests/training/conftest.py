import pytest


@pytest.fixture(scope="module")
def summary():
    from tests.training.helpers import build_training_update_summary

    return build_training_update_summary()
