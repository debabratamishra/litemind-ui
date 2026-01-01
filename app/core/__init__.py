"""
Core application utilities package.
"""
from .environment import (
    EnvironmentDetector,
    environment,
    is_containerized,
    is_docker,
    get_platform,
)

__all__ = [
    'EnvironmentDetector',
    'environment',
    'is_containerized',
    'is_docker',
    'get_platform',
]
