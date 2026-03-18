import os

DELFT_PROJECT_DIR = os.path.dirname(__file__)


def get_registry_path():
    env_path = os.environ.get("DELFT_REGISTRY_PATH")
    if env_path:
        return env_path
    return os.path.join(DELFT_PROJECT_DIR, "resources-registry.json")
