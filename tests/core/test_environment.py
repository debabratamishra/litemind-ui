"""
Unit tests for ``app.core.environment``.

The module exposes a singleton ``EnvironmentDetector`` plus module-level
convenience functions ``is_containerized()``, ``is_docker()`` and
``get_platform()``.

Detection is driven by ``EnvironmentDetector._detect_container_environment()``
(an ``lru_cache``-decorated method) which is invoked once in ``__init__`` and
stored on ``self._is_containerized``. The ``is_docker`` / ``is_containerized``
properties are aliases for that cached flag, and ``platform_name`` is set from
``platform.system()`` at init time.

Because the detector is a singleton with a class-level ``_initialized`` guard,
every test that needs a *controlled* instance resets the singleton state and
clears the method cache before constructing a fresh detector.
"""

import sys
from unittest.mock import patch

import pytest

from app.core.environment import (
    EnvironmentDetector,
    get_platform,
    is_containerized,
    is_docker,
)

# NOTE: ``app.core.__init__`` re-exports the singleton instance under the name
# ``environment``, which shadows the submodule attribute ``app.core.environment``.
# To rebind the module-level singleton (so the module functions observe a
# controlled detector) we must use the real module object from sys.modules.
_env_module = sys.modules["app.core.environment"]


@pytest.fixture
def fresh_detector():
    """Reset the singleton so a new ``EnvironmentDetector`` re-runs detection.

    Restores the original state afterwards so other tests are unaffected.
    """
    saved_instance = EnvironmentDetector._instance
    saved_initialized = EnvironmentDetector._initialized
    saved_singleton_var = _env_module.environment  # module-level singleton instance

    EnvironmentDetector._instance = None
    EnvironmentDetector._initialized = False
    # Clear the lru_cache so a real (env/file) detection run is not stale.
    EnvironmentDetector._detect_container_environment.cache_clear()

    yield

    EnvironmentDetector._instance = saved_instance
    EnvironmentDetector._initialized = saved_initialized
    _env_module.environment = saved_singleton_var
    EnvironmentDetector._detect_container_environment.cache_clear()


def test_get_platform_returns_current():
    """``get_platform()`` returns a recognized platform string."""
    plat = get_platform()
    assert isinstance(plat, str)
    assert plat in ("darwin", "linux", "windows", sys.platform.lower())


def test_get_platform_returns_controlled_value(fresh_detector):
    """``get_platform()`` reflects ``platform.system()`` via the property."""
    with patch("app.core.environment.platform.system", return_value="Linux"):
        EnvironmentDetector._instance = None
        EnvironmentDetector._initialized = False
        det = EnvironmentDetector()
        # Rebind the module-level singleton so the module function is exercised.
        _env_module.environment = det
        assert det.platform_name == "linux"
        assert get_platform() == "linux"


def test_is_docker_true_when_detected(fresh_detector):
    """Docker detection is True when the underlying source indicates docker."""
    with patch.object(
        EnvironmentDetector, "_detect_container_environment", return_value=True
    ):
        det = EnvironmentDetector()
        assert det.is_docker is True
        assert det.is_containerized is True


def test_is_docker_false_when_not_detected(fresh_detector):
    """Docker detection is False when the underlying source does not indicate it."""
    with patch.object(
        EnvironmentDetector, "_detect_container_environment", return_value=False
    ):
        det = EnvironmentDetector()
        assert det.is_docker is False
        assert det.is_containerized is False


def test_module_level_is_docker_mirrors_singleton(fresh_detector):
    """Module-level ``is_docker()`` / ``is_containerized()`` track the detector."""
    with patch.object(
        EnvironmentDetector, "_detect_container_environment", return_value=True
    ):
        det = EnvironmentDetector()
        _env_module.environment = det
        assert is_docker() is True
        assert is_containerized() is True


def test_is_containerized_true_via_container_env(mock_env, fresh_detector):
    """A container-specific env var (real check) flips detection to True.

    Uses the real ``_detect_container_environment`` with no filesystem access.
    """
    mock_env(DOCKER_CONTAINER="1")
    det = EnvironmentDetector()
    assert det.is_containerized is True
    assert det.is_docker is True


def test_is_containerized_true_via_kubernetes_env(mock_env, fresh_detector):
    """The Kubernetes service-host env var is also treated as containerized."""
    mock_env(KUBERNETES_SERVICE_HOST="10.0.0.1")
    det = EnvironmentDetector()
    assert det.is_containerized is True


def test_is_containerized_false_without_signals(mock_env, fresh_detector):
    """With no docker file, cgroup, or env signal, detection is False."""
    # ensure no container env vars leak through
    for var in ("DOCKER_CONTAINER", "CONTAINER", "KUBERNETES_SERVICE_HOST", "CONTAINER_NAME"):
        mock_env(**{var: ""})
    with patch("app.core.environment.Path") as mock_path:
        # No /.dockerenv, no /proc/1/cgroup, no container mounts.
        mock_path.return_value.exists.return_value = False
        det = EnvironmentDetector()
        assert det.is_containerized is False
        assert det.is_docker is False
