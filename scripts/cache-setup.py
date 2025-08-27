#!/usr/bin/env python3
"""
Cache Directory Setup Script

This script ensures that all required cache directories exist with proper permissions
before starting Docker containers. It handles OS-independent cache directory creation
for Huggingface models, Ollama cache, and application data directories.

Usage:
    python scripts/cache-setup.py [--check-only] [--verbose]
    
Options:
    --check-only    Only check if directories exist, don't create them
    --verbose       Enable verbose logging
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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def check_directories(host_paths: dict) -> dict:
    """
    Check if directories exist and are accessible.
    
    Args:
        host_paths: Dictionary of directory names to paths
        
    Returns:
        Dictionary mapping directory names to status info
    """
    results = {}
    
    for name, path_str in host_paths.items():
        path = Path(path_str)
        status = {
            'exists': path.exists(),
            'is_dir': path.is_dir() if path.exists() else False,
            'readable': False,
            'writable': False,
            'path': str(path)
        }
        
        if path.exists() and path.is_dir():
            try:
                # Test read access
                list(path.iterdir())
                status['readable'] = True
            except (PermissionError, OSError):
                pass
            
            try:
                # Test write access by creating a temporary file
                test_file = path / '.cache_setup_test'
                test_file.touch()
                test_file.unlink()
                status['writable'] = True
            except (PermissionError, OSError):
                pass
        
        results[name] = status
    
    return results


def print_directory_status(results: dict, verbose: bool = False):
    """Print directory status in a readable format."""
    print("\nDirectory Status:")
    print("=" * 60)
    
    for name, status in results.items():
        path = status['path']
        exists = "✓" if status['exists'] else "✗"
        readable = "✓" if status['readable'] else "✗"
        writable = "✓" if status['writable'] else "✗"
        
        print(f"{name:20} | {exists} Exists | {readable} Read | {writable} Write")
        if verbose:
            print(f"{'':20} | Path: {path}")
    
    print("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Set up cache directories for Docker deployment"
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check if directories exist, do not create them'
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
        # Get host cache paths
        host_paths = host_service_manager.get_host_cache_paths()
        
        print("Cache Directory Setup")
        print("=" * 60)
        print(f"Operating System: {host_service_manager.environment_config.is_containerized and 'Container' or 'Native'}")
        print(f"Platform: {sys.platform}")
        
        if args.check_only:
            print("\nChecking existing directories...")
            results = check_directories(host_paths)
            print_directory_status(results, args.verbose)
            
            # Check if all directories are ready
            all_ready = all(
                status['exists'] and status['readable'] and status['writable']
                for status in results.values()
            )
            
            if all_ready:
                print("\n✓ All directories are ready for Docker deployment!")
                sys.exit(0)
            else:
                print("\n✗ Some directories need to be created or have permission issues.")
                print("Run without --check-only to create missing directories.")
                sys.exit(1)
        
        else:
            print("\nCreating cache directories...")
            
            # Create directories
            creation_results = host_service_manager.ensure_host_cache_directories_exist()
            
            # Check final status
            final_results = check_directories(host_paths)
            print_directory_status(final_results, args.verbose)
            
            # Report results
            success_count = sum(1 for success in creation_results.values() if success)
            total_count = len(creation_results)
            
            print(f"\nCreation Results: {success_count}/{total_count} directories created successfully")
            
            if success_count == total_count:
                print("✓ All cache directories are ready for Docker deployment!")
                sys.exit(0)
            else:
                print("✗ Some directories could not be created. Check permissions and try again.")
                for name, success in creation_results.items():
                    if not success:
                        path = host_paths[name]
                        print(f"  Failed: {name} -> {path}")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during cache setup: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()