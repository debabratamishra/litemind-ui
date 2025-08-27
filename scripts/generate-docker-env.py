#!/usr/bin/env python3
"""
Docker Environment Generator

This script generates OS-independent environment variables for Docker Compose
to ensure proper cache directory mounting across different operating systems.

Usage:
    python scripts/generate-docker-env.py [--output .env.docker] [--verbose]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.services.host_service_manager import host_service_manager
except ImportError as e:
    print(f"Error importing host service manager: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def generate_docker_env(output_file: str = ".env.docker"):
    """
    Generate Docker environment file with OS-independent cache paths.
    
    Args:
        output_file: Path to the output environment file
    """
    logger = logging.getLogger(__name__)
    
    # Get host cache paths
    host_paths = host_service_manager.get_host_cache_paths()
    
    # Generate environment variables
    env_vars = {
        'HOST_HF_CACHE': host_paths['huggingface_cache'],
        'HOST_OLLAMA_CACHE': host_paths['ollama_cache'],
        'HOST_UPLOADS_DIR': host_paths['uploads'],
        'HOST_CHROMA_DB_DIR': host_paths['chroma_db'],
        'HOST_STORAGE_DIR': host_paths['storage'],
        'HOST_STREAMLIT_CONFIG_DIR': host_paths['streamlit_config'],
        # Container mount points (these are fixed)
        'CONTAINER_HF_CACHE': '/root/.cache/huggingface',
        'CONTAINER_OLLAMA_CACHE': '/root/.ollama',
        'CONTAINER_UPLOADS_DIR': '/app/uploads',
        'CONTAINER_CHROMA_DB_DIR': '/app/chroma_db',
        'CONTAINER_STORAGE_DIR': '/app/storage',
        'CONTAINER_STREAMLIT_CONFIG_DIR': '/app/.streamlit'
    }
    
    # Write environment file
    output_path = Path(output_file)
    
    with open(output_path, 'w') as f:
        f.write("# Docker Environment Variables for Cache Directory Management\n")
        f.write("# Generated automatically - do not edit manually\n")
        f.write(f"# Generated for platform: {sys.platform}\n\n")
        
        f.write("# Host system cache directories\n")
        for key, value in env_vars.items():
            if key.startswith('HOST_'):
                f.write(f"{key}={value}\n")
        
        f.write("\n# Container mount points\n")
        for key, value in env_vars.items():
            if key.startswith('CONTAINER_'):
                f.write(f"{key}={value}\n")
    
    logger.info(f"Generated Docker environment file: {output_path}")
    return env_vars


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate OS-independent Docker environment variables"
    )
    parser.add_argument(
        '--output',
        default='.env.docker',
        help='Output file path (default: .env.docker)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        print("Docker Environment Generator")
        print("=" * 50)
        print(f"Platform: {sys.platform}")
        print(f"Output file: {args.output}")
        
        # Generate environment variables
        env_vars = generate_docker_env(args.output)
        
        print("\nGenerated Environment Variables:")
        print("-" * 50)
        for key, value in env_vars.items():
            print(f"{key}={value}")
        
        print(f"\n✓ Environment file created: {args.output}")
        print("\nTo use with Docker Compose:")
        print(f"  docker-compose --env-file {args.output} up")
        
    except Exception as e:
        logger.error(f"Error generating Docker environment: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()