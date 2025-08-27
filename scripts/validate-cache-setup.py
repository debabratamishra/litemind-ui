#!/usr/bin/env python3
"""
Cache Setup Validation Script

This script validates that the cache directory setup is working correctly
by testing directory access, permissions, and Docker environment configuration.

Usage:
    python scripts/validate-cache-setup.py [--docker-test] [--verbose]
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.services.host_service_manager import host_service_manager
    from config import Config
except ImportError as e:
    print(f"Error importing modules: {e}")
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


def test_directory_access(path: Path, name: str) -> dict:
    """Test directory access and permissions."""
    result = {
        'name': name,
        'path': str(path),
        'exists': False,
        'readable': False,
        'writable': False,
        'error': None
    }
    
    try:
        result['exists'] = path.exists()
        
        if result['exists']:
            # Test read access
            try:
                list(path.iterdir())
                result['readable'] = True
            except (PermissionError, OSError) as e:
                result['error'] = f"Read error: {e}"
            
            # Test write access
            try:
                test_file = path / f'.test_write_{os.getpid()}'
                test_file.write_text('test')
                test_file.unlink()
                result['writable'] = True
            except (PermissionError, OSError) as e:
                if not result['error']:
                    result['error'] = f"Write error: {e}"
                else:
                    result['error'] += f", Write error: {e}"
        else:
            result['error'] = "Directory does not exist"
            
    except Exception as e:
        result['error'] = f"Unexpected error: {e}"
    
    return result


def test_cache_directories():
    """Test all cache directories."""
    logger = logging.getLogger(__name__)
    
    print("Testing Cache Directory Access")
    print("=" * 50)
    
    # Get configuration
    config = Config.get_dynamic_config()
    
    # Test directories
    directories = {
        'HF Cache': Path(config['hf_cache_dir']),
        'Ollama Cache': Path(config['ollama_cache_dir']),
        'Uploads': Path(config['upload_dir']),
        'ChromaDB': Path(config['chroma_db_dir'])
    }
    
    results = []
    all_passed = True
    
    for name, path in directories.items():
        result = test_directory_access(path, name)
        results.append(result)
        
        status = "✓" if (result['exists'] and result['readable'] and result['writable']) else "✗"
        print(f"{status} {name:12} | {result['path']}")
        
        if result['error']:
            print(f"  Error: {result['error']}")
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("✓ All cache directories are accessible!")
        logger.info("Cache directory validation passed")
    else:
        print("✗ Some cache directories have issues")
        logger.error("Cache directory validation failed")
    
    return all_passed, results


def test_docker_environment():
    """Test Docker environment configuration."""
    logger = logging.getLogger(__name__)
    
    print("\nTesting Docker Environment Configuration")
    print("=" * 50)
    
    env_file = Path('.env.docker')
    
    if not env_file.exists():
        print("✗ .env.docker file not found")
        print("  Run: python scripts/generate-docker-env.py")
        return False
    
    # Read environment variables
    env_vars = {}
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    except Exception as e:
        print(f"✗ Error reading .env.docker: {e}")
        return False
    
    print(f"✓ Found {len(env_vars)} environment variables")
    
    # Check required variables
    required_vars = [
        'HOST_HF_CACHE',
        'HOST_OLLAMA_CACHE',
        'CONTAINER_HF_CACHE',
        'CONTAINER_OLLAMA_CACHE'
    ]
    
    missing_vars = []
    for var in required_vars:
        if var not in env_vars:
            missing_vars.append(var)
        else:
            print(f"✓ {var}={env_vars[var]}")
    
    if missing_vars:
        print(f"✗ Missing variables: {', '.join(missing_vars)}")
        return False
    
    # Validate host paths exist
    host_paths = {
        'HF Cache': env_vars.get('HOST_HF_CACHE'),
        'Ollama Cache': env_vars.get('HOST_OLLAMA_CACHE')
    }
    
    for name, path in host_paths.items():
        if path and Path(path).exists():
            print(f"✓ {name} path exists: {path}")
        else:
            print(f"✗ {name} path missing: {path}")
            return False
    
    print("✓ Docker environment configuration is valid")
    logger.info("Docker environment validation passed")
    return True


def test_host_service_manager():
    """Test host service manager functionality."""
    logger = logging.getLogger(__name__)
    
    print("\nTesting Host Service Manager")
    print("=" * 50)
    
    try:
        # Test environment detection
        is_containerized = host_service_manager.is_containerized
        print(f"✓ Environment detection: {'Container' if is_containerized else 'Native'}")
        
        # Test cache path generation
        host_paths = host_service_manager.get_host_cache_paths()
        print(f"✓ Generated {len(host_paths)} host cache paths")
        
        # Test directory creation
        creation_results = host_service_manager.ensure_host_cache_directories_exist()
        success_count = sum(1 for success in creation_results.values() if success)
        total_count = len(creation_results)
        
        if success_count == total_count:
            print(f"✓ Directory creation: {success_count}/{total_count} successful")
        else:
            print(f"✗ Directory creation: {success_count}/{total_count} successful")
            return False
        
        # Test dynamic configuration
        dynamic_config = host_service_manager.get_dynamic_config()
        required_keys = ['is_containerized', 'hf_cache_dir', 'ollama_cache_dir']
        
        for key in required_keys:
            if key in dynamic_config:
                print(f"✓ Dynamic config has {key}")
            else:
                print(f"✗ Dynamic config missing {key}")
                return False
        
        print("✓ Host service manager is working correctly")
        logger.info("Host service manager validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Host service manager error: {e}")
        logger.error(f"Host service manager validation failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate cache directory setup"
    )
    parser.add_argument(
        '--docker-test',
        action='store_true',
        help='Include Docker environment tests'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    print("Cache Setup Validation")
    print("=" * 50)
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    
    all_tests_passed = True
    
    try:
        # Test 1: Cache directories
        cache_passed, _ = test_cache_directories()
        all_tests_passed = all_tests_passed and cache_passed
        
        # Test 2: Host service manager
        hsm_passed = test_host_service_manager()
        all_tests_passed = all_tests_passed and hsm_passed
        
        # Test 3: Docker environment (optional)
        if args.docker_test:
            docker_passed = test_docker_environment()
            all_tests_passed = all_tests_passed and docker_passed
        
        print("\n" + "=" * 50)
        
        if all_tests_passed:
            print("🎉 All validation tests passed!")
            print("✓ Cache directory setup is working correctly")
            sys.exit(0)
        else:
            print("❌ Some validation tests failed")
            print("✗ Please check the errors above and run setup again")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()