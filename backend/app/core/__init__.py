"""
Core application utilities package.
"""
from .environment import (
    EnvironmentDetector,
    environment,
    get_platform,
    is_containerized,
    is_docker,
)

__all__ = [
    'EnvironmentDetector',
    'environment',
    'is_containerized',
    'is_docker',
    'get_platform',
]
