#!/usr/bin/env python3
"""
Docker Startup Validation Script for LLMWebUI

This script validates that all required host services are available and
the container environment is properly configured before starting the application.
It performs comprehensive checks for:
- Host service connectivity (Ollama, vLLM)
- Volume mount accessibility
- Cache directory permissions
- Environment variable validation
"""

import sys
import os
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class StartupValidator:
    """Validates container startup requirements and host service connectivity."""
    
    def __init__(self):
        self.validation_results = {
            "environment": {},
            "services": {},
            "volumes": {},
            "permissions": {},
            "overall_status": "unknown"
        }
        
    def detect_container_environment(self) -> bool:
        """Detect if running inside a container."""
        container_indicators = [
            Path('/.dockerenv').exists(),
            os.getenv('CONTAINER') is not None,
            os.getenv('DOCKER_CONTAINER') is not None
        ]
        
        # Check cgroup for container indicators
        try:
            with open('/proc/1/cgroup', 'r') as f:
                cgroup_content = f.read()
                if 'docker' in cgroup_content or 'containerd' in cgroup_content:
                    container_indicators.append(True)
        except (FileNotFoundError, PermissionError):
            pass
            
        return any(container_indicators)
    
    def validate_environment_variables(self) -> Dict[str, bool]:
        """Validate required environment variables are set."""
        required_vars = [
            'OLLAMA_API_URL',
            'VLLM_API_URL'
        ]
        
        optional_vars = [
            'HF_HOME',
            'OLLAMA_MODELS',
            'UPLOAD_FOLDER',
            'CHROMA_DB_PATH',
            'STORAGE_PATH'
        ]
        
        results = {}
        
        # Check required variables
        for var in required_vars:
            value = os.getenv(var)
            results[var] = {
                'present': value is not None,
                'value': value if value else 'NOT SET',
                'required': True
            }
            
        # Check optional variables
        for var in optional_vars:
            value = os.getenv(var)
            results[var] = {
                'present': value is not None,
                'value': value if value else 'using default',
                'required': False
            }
            
        return results
    
    def validate_volume_mounts(self) -> Dict[str, Dict[str, any]]:
        """Validate that volume mounts are accessible and have proper permissions."""
        # Expected volume mount points in container
        volume_mounts = {
            'huggingface_cache': os.getenv('HF_HOME', '/root/.cache/huggingface'),
            'ollama_cache': os.getenv('OLLAMA_MODELS', '/root/.ollama'),
            'uploads': os.getenv('UPLOAD_FOLDER', '/app/uploads'),
            'chroma_db': os.getenv('CHROMA_DB_PATH', '/app/chroma_db'),
            'storage': os.getenv('STORAGE_PATH', '/app/storage'),
            'streamlit_config': '/app/.streamlit'
        }
        
        results = {}
        
        for name, path_str in volume_mounts.items():
            path = Path(path_str)
            result = {
                'path': str(path),
                'exists': path.exists(),
                'is_directory': path.is_dir() if path.exists() else False,
                'readable': False,
                'writable': False,
                'permissions': None
            }
            
            if path.exists():
                try:
                    # Test read access
                    result['readable'] = os.access(path, os.R_OK)
                    
                    # Test write access
                    result['writable'] = os.access(path, os.W_OK)
                    
                    # Get permissions
                    stat_info = path.stat()
                    result['permissions'] = oct(stat_info.st_mode)[-3:]
                    
                except Exception as e:
                    result['error'] = str(e)
            else:
                # Try to create directory if it doesn't exist
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    os.chmod(path, 0o755)
                    result['exists'] = True
                    result['is_directory'] = True
                    result['readable'] = True
                    result['writable'] = True
                    result['permissions'] = '755'
                    result['created'] = True
                except Exception as e:
                    result['creation_error'] = str(e)
            
            results[name] = result
            
        return results
    
    async def validate_service_connectivity(self, service_name: str, url: str, timeout: float = 10.0) -> Dict[str, any]:
        """Validate connectivity to a host service with detailed diagnostics."""
        result = {
            'name': service_name,
            'url': url,
            'available': False,
            'response_time_ms': None,
            'status_code': None,
            'error': None,
            'details': {}
        }
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Determine appropriate endpoint based on service
                if '11434' in url:  # Ollama
                    test_url = f"{url}/api/tags"
                    result['service_type'] = 'ollama'
                elif '8001' in url:  # vLLM
                    test_url = f"{url}/v1/models"
                    result['service_type'] = 'vllm'
                else:
                    test_url = f"{url}/health"
                    result['service_type'] = 'generic'
                
                logger.info(f"Testing {service_name} connectivity at {test_url}")
                
                response = await client.get(test_url)
                response_time = (time.time() - start_time) * 1000
                
                result['available'] = response.status_code == 200
                result['status_code'] = response.status_code
                result['response_time_ms'] = round(response_time, 2)
                
                if response.status_code == 200:
                    # Try to parse response for additional details
                    try:
                        response_data = response.json()
                        if service_name.lower() == 'ollama' and 'models' in response_data:
                            result['details']['model_count'] = len(response_data['models'])
                            result['details']['models'] = [m.get('name', 'unknown') for m in response_data['models'][:5]]
                        elif service_name.lower() == 'vllm' and 'data' in response_data:
                            result['details']['model_count'] = len(response_data['data'])
                    except Exception:
                        pass  # Response parsing is optional
                        
                else:
                    result['error'] = f"HTTP {response.status_code}"
                    
        except httpx.TimeoutException:
            result['error'] = f"Connection timeout after {timeout}s"
            result['response_time_ms'] = timeout * 1000
            
        except httpx.ConnectError as e:
            result['error'] = f"Connection refused: {str(e)}"
            
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
        
        return result
    
    async def validate_all_services(self) -> Dict[str, Dict[str, any]]:
        """Validate all required host services."""
        services = {
            'ollama': os.getenv('OLLAMA_API_URL', 'http://localhost:11434'),
            'vllm': os.getenv('VLLM_API_URL', 'http://localhost:8001')
        }
        
        results = {}
        
        # Test services concurrently with longer timeout for startup
        tasks = [
            self.validate_service_connectivity(name, url, timeout=15.0)
            for name, url in services.items()
        ]
        
        service_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (name, url), result in zip(services.items(), service_results):
            if isinstance(result, Exception):
                results[name] = {
                    'name': name,
                    'url': url,
                    'available': False,
                    'error': str(result)
                }
            else:
                results[name] = result
        
        return results
    
    def check_critical_requirements(self) -> Tuple[bool, List[str]]:
        """Check if critical requirements are met for application startup."""
        errors = []
        
        # Check if we're in a container (expected for this validation)
        if not self.detect_container_environment():
            errors.append("Not running in container environment")
        
        # Check volume mounts
        volume_results = self.validation_results.get('volumes', {})
        critical_volumes = ['uploads', 'chroma_db', 'storage']
        
        for vol_name in critical_volumes:
            vol_info = volume_results.get(vol_name, {})
            if not vol_info.get('exists', False):
                errors.append(f"Critical volume mount missing: {vol_name}")
            elif not vol_info.get('writable', False):
                errors.append(f"Critical volume not writable: {vol_name}")
        
        # Check environment variables
        env_results = self.validation_results.get('environment', {})
        required_env_vars = ['OLLAMA_API_URL', 'VLLM_API_URL']
        
        for var in required_env_vars:
            var_info = env_results.get(var, {})
            if not var_info.get('present', False):
                errors.append(f"Required environment variable missing: {var}")
        
        return len(errors) == 0, errors
    
    async def run_full_validation(self) -> Dict[str, any]:
        """Run complete startup validation."""
        logger.info("🚀 Starting Docker container validation...")
        
        # Environment validation
        logger.info("📋 Validating environment variables...")
        self.validation_results['environment'] = self.validate_environment_variables()
        
        # Volume mount validation
        logger.info("💾 Validating volume mounts...")
        self.validation_results['volumes'] = self.validate_volume_mounts()
        
        # Service connectivity validation
        logger.info("🌐 Validating host service connectivity...")
        self.validation_results['services'] = await self.validate_all_services()
        
        # Check critical requirements
        logger.info("🔍 Checking critical requirements...")
        is_ready, errors = self.check_critical_requirements()
        
        self.validation_results['overall_status'] = 'ready' if is_ready else 'failed'
        self.validation_results['critical_errors'] = errors
        self.validation_results['timestamp'] = time.time()
        
        return self.validation_results
    
    def print_validation_report(self):
        """Print a human-readable validation report."""
        print("\n" + "="*60)
        print("🐳 DOCKER CONTAINER STARTUP VALIDATION REPORT")
        print("="*60)
        
        # Environment Variables
        print("\n📋 Environment Variables:")
        env_results = self.validation_results.get('environment', {})
        for var, info in env_results.items():
            status = "✅" if info['present'] else ("❌" if info['required'] else "⚠️")
            print(f"  {status} {var}: {info['value']}")
        
        # Volume Mounts
        print("\n💾 Volume Mounts:")
        volume_results = self.validation_results.get('volumes', {})
        for name, info in volume_results.items():
            if info.get('exists', False) and info.get('writable', False):
                status = "✅"
            elif info.get('exists', False):
                status = "⚠️"
            else:
                status = "❌"
            
            print(f"  {status} {name}: {info['path']}")
            if info.get('permissions'):
                print(f"      Permissions: {info['permissions']}")
            if info.get('error'):
                print(f"      Error: {info['error']}")
        
        # Host Services
        print("\n🌐 Host Services:")
        service_results = self.validation_results.get('services', {})
        for name, info in service_results.items():
            status = "✅" if info.get('available', False) else "❌"
            print(f"  {status} {name.upper()}: {info['url']}")
            
            if info.get('available'):
                print(f"      Response time: {info.get('response_time_ms', 0):.1f}ms")
                if info.get('details', {}).get('model_count'):
                    print(f"      Models available: {info['details']['model_count']}")
            else:
                print(f"      Error: {info.get('error', 'Unknown error')}")
        
        # Overall Status
        print(f"\n🎯 Overall Status: {self.validation_results['overall_status'].upper()}")
        
        if self.validation_results.get('critical_errors'):
            print("\n❌ Critical Errors:")
            for error in self.validation_results['critical_errors']:
                print(f"  - {error}")
        
        print("\n" + "="*60)


async def main():
    """Main startup validation routine."""
    validator = StartupValidator()
    
    try:
        # Run validation
        results = await validator.run_full_validation()
        
        # Print report
        validator.print_validation_report()
        
        # Save results to file for other processes to read
        results_file = Path('/tmp/startup_validation.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Validation results saved to {results_file}")
        except Exception as e:
            logger.warning(f"Could not save validation results: {e}")
        
        # Exit with appropriate code
        if results['overall_status'] == 'ready':
            logger.info("✅ Container startup validation PASSED")
            
            # Check if any services are unavailable but not critical
            service_warnings = []
            for name, info in results.get('services', {}).items():
                if not info.get('available', False):
                    service_warnings.append(f"{name.upper()} service unavailable")
            
            if service_warnings:
                logger.warning("⚠️  Some optional services are unavailable:")
                for warning in service_warnings:
                    logger.warning(f"   - {warning}")
                logger.info("Application can start but some features may be limited")
            
            sys.exit(0)
        else:
            logger.error("❌ Container startup validation FAILED")
            logger.error("Critical errors prevent application startup")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())