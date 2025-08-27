#!/usr/bin/env python3
"""
Enhanced Health Check Script for LLMWebUI Docker Containers

This script provides comprehensive health checking capabilities for Docker containers,
including startup, readiness, liveness, and host service connectivity checks.
It can be used as a Docker HEALTHCHECK command or run standalone for diagnostics.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import subprocess

try:
    import requests
except ImportError:
    print("Warning: requests library not available, using basic HTTP checks")
    requests = None


class HealthChecker:
    """Comprehensive health checker for LLMWebUI Docker containers."""
    
    def __init__(self, timeout: int = 10, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.results = {
            "timestamp": time.time(),
            "checks": {},
            "overall_status": "unknown"
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def check_service_http(self, name: str, url: str, timeout: Optional[int] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if an HTTP service is healthy."""
        if timeout is None:
            timeout = self.timeout
            
        details = {"url": url, "timeout": timeout}
        
        if requests is None:
            # Fallback to curl if requests is not available
            return self._check_service_curl(name, url, timeout)
        
        try:
            self.log(f"Checking {name} at {url}")
            start_time = time.time()
            
            response = requests.get(url, timeout=timeout)
            response_time = (time.time() - start_time) * 1000
            
            details.update({
                "status_code": response.status_code,
                "response_time_ms": round(response_time, 2)
            })
            
            if response.status_code == 200:
                # Try to parse JSON response for additional details
                try:
                    response_data = response.json()
                    details["response_data"] = response_data
                except:
                    details["response_text"] = response.text[:200]  # First 200 chars
                
                return True, f"✅ {name} is healthy", details
            else:
                return False, f"❌ {name} returned HTTP {response.status_code}", details
                
        except requests.exceptions.ConnectionError:
            details["error"] = "Connection refused"
            return False, f"❌ {name} is not reachable", details
        except requests.exceptions.Timeout:
            details["error"] = "Timeout"
            return False, f"❌ {name} timed out", details
        except Exception as e:
            details["error"] = str(e)
            return False, f"❌ {name} error: {str(e)}", details
    
    def _check_service_curl(self, name: str, url: str, timeout: int) -> Tuple[bool, str, Dict[str, Any]]:
        """Fallback HTTP check using curl."""
        details = {"url": url, "timeout": timeout, "method": "curl"}
        
        try:
            result = subprocess.run([
                'curl', '-s', '-f', '--max-time', str(timeout), url
            ], capture_output=True, text=True, timeout=timeout + 2)
            
            if result.returncode == 0:
                details["curl_output"] = result.stdout[:200]
                return True, f"✅ {name} is healthy (curl)", details
            else:
                details["curl_error"] = result.stderr
                return False, f"❌ {name} failed (curl exit {result.returncode})", details
                
        except subprocess.TimeoutExpired:
            details["error"] = "curl timeout"
            return False, f"❌ {name} timed out (curl)", details
        except FileNotFoundError:
            details["error"] = "curl not available"
            return False, f"❌ {name} check failed (curl not found)", details
        except Exception as e:
            details["error"] = str(e)
            return False, f"❌ {name} error (curl): {str(e)}", details
    
    def check_container_services(self) -> Dict[str, Dict[str, Any]]:
        """Check containerized services."""
        services = {
            "fastapi_health": "http://localhost:8000/health",
            "fastapi_ready": "http://localhost:8000/health/ready", 
            "fastapi_live": "http://localhost:8000/health/live",
            "streamlit": "http://localhost:8501/_stcore/health"
        }
        
        results = {}
        
        for service_name, url in services.items():
            is_healthy, message, details = self.check_service_http(service_name, url)
            results[service_name] = {
                "healthy": is_healthy,
                "message": message,
                "details": details
            }
        
        return results
    
    def check_host_services(self) -> Dict[str, Dict[str, Any]]:
        """Check host services that containers depend on."""
        # Get service URLs from environment or use defaults
        ollama_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
        vllm_url = os.getenv('VLLM_API_URL', 'http://localhost:8001')
        
        services = {
            "ollama": f"{ollama_url}/api/tags",
            "vllm": f"{vllm_url}/v1/models"
        }
        
        results = {}
        
        for service_name, url in services.items():
            is_healthy, message, details = self.check_service_http(service_name, url, timeout=5)
            results[service_name] = {
                "healthy": is_healthy,
                "message": message,
                "details": details,
                "optional": service_name == "vllm"  # vLLM is optional
            }
        
        return results
    
    def check_filesystem_health(self) -> Dict[str, Dict[str, Any]]:
        """Check filesystem and volume mount health."""
        # Critical directories that should be accessible
        critical_dirs = [
            "/app/uploads",
            "/app/chroma_db", 
            "/app/storage",
            "/root/.cache/huggingface",
            "/root/.ollama"
        ]
        
        results = {}
        
        for dir_path in critical_dirs:
            path = Path(dir_path)
            dir_name = path.name
            
            result = {
                "path": str(path),
                "exists": path.exists(),
                "is_directory": False,
                "readable": False,
                "writable": False
            }
            
            if path.exists():
                result["is_directory"] = path.is_dir()
                if path.is_dir():
                    result["readable"] = os.access(path, os.R_OK)
                    result["writable"] = os.access(path, os.W_OK)
                    
                    # Test actual write capability
                    try:
                        test_file = path / ".health_check_test"
                        test_file.write_text("test")
                        test_file.unlink()
                        result["write_test"] = True
                    except Exception as e:
                        result["write_test"] = False
                        result["write_error"] = str(e)
            
            # Determine if this directory check passed
            is_healthy = (result["exists"] and result["is_directory"] and 
                         result["readable"] and result["writable"])
            
            results[dir_name] = {
                "healthy": is_healthy,
                "message": f"{'✅' if is_healthy else '❌'} {dir_name}: {dir_path}",
                "details": result
            }
        
        return results
    
    def check_process_health(self) -> Dict[str, Dict[str, Any]]:
        """Check process-level health indicators."""
        results = {}
        
        # Check if main process is responsive
        try:
            pid = os.getpid()
            results["main_process"] = {
                "healthy": True,
                "message": f"✅ Main process running (PID: {pid})",
                "details": {"pid": pid}
            }
        except Exception as e:
            results["main_process"] = {
                "healthy": False,
                "message": f"❌ Process check failed: {e}",
                "details": {"error": str(e)}
            }
        
        # Check memory usage if psutil is available
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            memory_ok = memory.percent < 90
            disk_ok = disk.percent < 95
            
            results["resources"] = {
                "healthy": memory_ok and disk_ok,
                "message": f"{'✅' if memory_ok and disk_ok else '⚠️'} Resources: Memory {memory.percent:.1f}%, Disk {disk.percent:.1f}%",
                "details": {
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            }
        except ImportError:
            results["resources"] = {
                "healthy": True,
                "message": "ℹ️  Resource monitoring not available (psutil not installed)",
                "details": {"psutil_available": False}
            }
        
        return results
    
    def run_startup_check(self) -> bool:
        """Run startup health check."""
        self.log("Running startup health check")
        
        # For startup, we mainly check that the container is properly initialized
        filesystem_results = self.check_filesystem_health()
        process_results = self.check_process_health()
        
        self.results["checks"]["filesystem"] = filesystem_results
        self.results["checks"]["process"] = process_results
        
        # Startup passes if filesystem and process checks pass
        filesystem_ok = all(check["healthy"] for check in filesystem_results.values())
        process_ok = all(check["healthy"] for check in process_results.values())
        
        startup_ok = filesystem_ok and process_ok
        self.results["overall_status"] = "started" if startup_ok else "starting"
        
        return startup_ok
    
    def run_readiness_check(self) -> bool:
        """Run readiness health check."""
        self.log("Running readiness health check")
        
        # For readiness, check that services are responding
        container_results = self.check_container_services()
        filesystem_results = self.check_filesystem_health()
        
        self.results["checks"]["container_services"] = container_results
        self.results["checks"]["filesystem"] = filesystem_results
        
        # Readiness passes if FastAPI health endpoint responds
        fastapi_ready = container_results.get("fastapi_health", {}).get("healthy", False)
        filesystem_ok = all(check["healthy"] for check in filesystem_results.values())
        
        ready = fastapi_ready and filesystem_ok
        self.results["overall_status"] = "ready" if ready else "not_ready"
        
        return ready
    
    def run_liveness_check(self) -> bool:
        """Run liveness health check."""
        self.log("Running liveness health check")
        
        # For liveness, check that the process is alive and responsive
        container_results = self.check_container_services()
        process_results = self.check_process_health()
        
        self.results["checks"]["container_services"] = container_results
        self.results["checks"]["process"] = process_results
        
        # Liveness passes if FastAPI liveness endpoint responds
        fastapi_live = container_results.get("fastapi_live", {}).get("healthy", False)
        process_ok = all(check["healthy"] for check in process_results.values())
        
        alive = fastapi_live and process_ok
        self.results["overall_status"] = "alive" if alive else "dead"
        
        return alive
    
    def run_comprehensive_check(self) -> bool:
        """Run comprehensive health check including host services."""
        self.log("Running comprehensive health check")
        
        # Check all components
        container_results = self.check_container_services()
        host_results = self.check_host_services()
        filesystem_results = self.check_filesystem_health()
        process_results = self.check_process_health()
        
        self.results["checks"]["container_services"] = container_results
        self.results["checks"]["host_services"] = host_results
        self.results["checks"]["filesystem"] = filesystem_results
        self.results["checks"]["process"] = process_results
        
        # Overall health calculation
        container_ok = container_results.get("fastapi_health", {}).get("healthy", False)
        filesystem_ok = all(check["healthy"] for check in filesystem_results.values())
        process_ok = all(check["healthy"] for check in process_results.values())
        
        # Host services - only Ollama is required, vLLM is optional
        ollama_ok = host_results.get("ollama", {}).get("healthy", False)
        
        overall_healthy = container_ok and filesystem_ok and process_ok and ollama_ok
        self.results["overall_status"] = "healthy" if overall_healthy else "unhealthy"
        
        return overall_healthy
    
    def print_results(self):
        """Print human-readable results."""
        print(f"\n🏥 LLMWebUI Health Check Results")
        print("=" * 50)
        print(f"Overall Status: {self.results['overall_status'].upper()}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.results['timestamp']))}")
        
        for category, checks in self.results["checks"].items():
            print(f"\n📋 {category.replace('_', ' ').title()}:")
            for check_name, check_result in checks.items():
                print(f"  {check_result['message']}")
                
                if self.verbose and "details" in check_result:
                    details = check_result["details"]
                    if isinstance(details, dict):
                        for key, value in details.items():
                            if key not in ["response_data", "response_text"]:
                                print(f"    {key}: {value}")
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            self.log(f"Results saved to {output_file}")
        except Exception as e:
            self.log(f"Failed to save results: {e}", "ERROR")


def main():
    """Main health check function."""
    parser = argparse.ArgumentParser(description='LLMWebUI Docker Health Check')
    parser.add_argument('--type', choices=['startup', 'readiness', 'liveness', 'comprehensive'], 
                       default='comprehensive', help='Type of health check to perform')
    parser.add_argument('--timeout', type=int, default=10, 
                       help='Timeout for HTTP requests (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--output', '-o', type=str, 
                       help='Save results to JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output (for Docker HEALTHCHECK)')
    
    args = parser.parse_args()
    
    # Create health checker
    checker = HealthChecker(timeout=args.timeout, verbose=args.verbose and not args.quiet)
    
    # Run appropriate check
    if args.type == 'startup':
        success = checker.run_startup_check()
    elif args.type == 'readiness':
        success = checker.run_readiness_check()
    elif args.type == 'liveness':
        success = checker.run_liveness_check()
    else:  # comprehensive
        success = checker.run_comprehensive_check()
    
    # Output results
    if not args.quiet:
        checker.print_results()
    
    # Save results if requested
    if args.output:
        checker.save_results(args.output)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()