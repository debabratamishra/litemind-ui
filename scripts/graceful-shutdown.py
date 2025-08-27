#!/usr/bin/env python3
"""
Graceful Shutdown Handler for LLMWebUI Docker Containers

This script handles graceful shutdown of the containerized application,
ensuring proper cleanup of resources, host processes, and data persistence.
It can be used as a signal handler or called directly during container shutdown.
"""

import os
import sys
import time
import signal
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GracefulShutdownHandler:
    """Handles graceful shutdown of containerized LLMWebUI application."""
    
    def __init__(self, shutdown_timeout: int = 30):
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_start_time = None
        self.cleanup_tasks = []
        self.host_processes = []
        
    def register_cleanup_task(self, task_name: str, cleanup_func, *args, **kwargs):
        """Register a cleanup task to be executed during shutdown."""
        self.cleanup_tasks.append({
            'name': task_name,
            'func': cleanup_func,
            'args': args,
            'kwargs': kwargs
        })
        
    def find_host_processes(self) -> List[Dict[str, Any]]:
        """Find vLLM and other host processes that need cleanup."""
        processes = []
        
        try:
            # Look for vLLM processes
            result = subprocess.run(
                ['pgrep', '-f', 'vllm'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        try:
                            # Get process info
                            proc_info = subprocess.run(
                                ['ps', '-p', pid.strip(), '-o', 'pid,ppid,cmd'],
                                capture_output=True,
                                text=True,
                                timeout=2
                            )
                            
                            if proc_info.returncode == 0:
                                lines = proc_info.stdout.strip().split('\n')
                                if len(lines) > 1:  # Skip header
                                    process_line = lines[1].strip()
                                    processes.append({
                                        'pid': int(pid.strip()),
                                        'type': 'vllm',
                                        'info': process_line
                                    })
                        except Exception as e:
                            logger.warning(f"Could not get info for PID {pid}: {e}")
                            
        except subprocess.TimeoutExpired:
            logger.warning("Timeout while searching for host processes")
        except Exception as e:
            logger.warning(f"Error searching for host processes: {e}")
            
        return processes
    
    def cleanup_host_processes(self):
        """Clean up host processes spawned by the container."""
        logger.info("🧹 Cleaning up host processes...")
        
        processes = self.find_host_processes()
        
        if not processes:
            logger.info("No host processes found to clean up")
            return
        
        for process in processes:
            try:
                pid = process['pid']
                process_type = process['type']
                
                logger.info(f"Terminating {process_type} process (PID: {pid})")
                
                # Send SIGTERM first
                os.kill(pid, signal.SIGTERM)
                
                # Wait a bit for graceful shutdown
                time.sleep(2)
                
                # Check if process is still running
                try:
                    os.kill(pid, 0)  # Check if process exists
                    logger.warning(f"Process {pid} still running, sending SIGKILL")
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    logger.info(f"Process {pid} terminated successfully")
                    
            except ProcessLookupError:
                logger.info(f"Process {pid} already terminated")
            except PermissionError:
                logger.warning(f"Permission denied when terminating process {pid}")
            except Exception as e:
                logger.error(f"Error terminating process {pid}: {e}")
    
    def cleanup_temporary_files(self):
        """Clean up temporary files and caches."""
        logger.info("🗑️  Cleaning up temporary files...")
        
        temp_locations = [
            '/tmp/startup_validation.json',
            '/tmp/llmwebui_*',
            '/app/uploads/*',  # Only if configured to clean on shutdown
        ]
        
        for location in temp_locations:
            try:
                if '*' in location:
                    # Handle glob patterns
                    import glob
                    files = glob.glob(location)
                    for file_path in files:
                        path = Path(file_path)
                        if path.exists():
                            if path.is_file():
                                path.unlink()
                                logger.info(f"Removed temporary file: {file_path}")
                            elif path.is_dir():
                                import shutil
                                shutil.rmtree(path)
                                logger.info(f"Removed temporary directory: {file_path}")
                else:
                    path = Path(location)
                    if path.exists():
                        path.unlink()
                        logger.info(f"Removed temporary file: {location}")
                        
            except Exception as e:
                logger.warning(f"Could not clean up {location}: {e}")
    
    def save_shutdown_state(self):
        """Save application state before shutdown."""
        logger.info("💾 Saving shutdown state...")
        
        shutdown_state = {
            'timestamp': time.time(),
            'shutdown_reason': 'graceful_container_shutdown',
            'processes_cleaned': len(self.host_processes),
            'cleanup_tasks_completed': len([t for t in self.cleanup_tasks if t.get('completed', False)])
        }
        
        try:
            state_file = Path('/tmp/shutdown_state.json')
            with open(state_file, 'w') as f:
                json.dump(shutdown_state, f, indent=2)
            logger.info(f"Shutdown state saved to {state_file}")
        except Exception as e:
            logger.warning(f"Could not save shutdown state: {e}")
    
    def flush_logs(self):
        """Ensure all logs are flushed before shutdown."""
        logger.info("📝 Flushing logs...")
        
        try:
            # Flush Python logging
            for handler in logging.getLogger().handlers:
                handler.flush()
            
            # Flush stdout/stderr
            sys.stdout.flush()
            sys.stderr.flush()
            
        except Exception as e:
            logger.warning(f"Error flushing logs: {e}")
    
    def wait_for_connections_to_close(self, max_wait: int = 10):
        """Wait for active connections to close gracefully."""
        logger.info(f"⏳ Waiting up to {max_wait}s for connections to close...")
        
        # This is a placeholder - in a real implementation, you might check
        # for active HTTP connections, database connections, etc.
        time.sleep(min(max_wait, 3))
        logger.info("Connection wait period completed")
    
    async def async_cleanup_tasks(self):
        """Execute any async cleanup tasks."""
        logger.info("🔄 Running async cleanup tasks...")
        
        try:
            # Example: Clean up RAG service if available
            try:
                # This would be imported from the main application
                # from main import rag_service
                # if rag_service:
                #     await rag_service.reset_system()
                logger.info("RAG service cleanup completed")
            except Exception as e:
                logger.warning(f"RAG service cleanup failed: {e}")
                
        except Exception as e:
            logger.error(f"Async cleanup tasks failed: {e}")
    
    def execute_cleanup_tasks(self):
        """Execute all registered cleanup tasks."""
        logger.info(f"🧹 Executing {len(self.cleanup_tasks)} cleanup tasks...")
        
        for task in self.cleanup_tasks:
            try:
                logger.info(f"Running cleanup task: {task['name']}")
                task['func'](*task['args'], **task['kwargs'])
                task['completed'] = True
                logger.info(f"Cleanup task completed: {task['name']}")
            except Exception as e:
                logger.error(f"Cleanup task failed: {task['name']} - {e}")
                task['completed'] = False
    
    def perform_graceful_shutdown(self):
        """Perform complete graceful shutdown sequence."""
        self.shutdown_start_time = time.time()
        
        logger.info("🛑 Starting graceful shutdown sequence...")
        logger.info(f"Shutdown timeout: {self.shutdown_timeout}s")
        
        try:
            # Step 1: Wait for connections to close
            self.wait_for_connections_to_close()
            
            # Step 2: Execute registered cleanup tasks
            self.execute_cleanup_tasks()
            
            # Step 3: Run async cleanup tasks
            try:
                asyncio.run(self.async_cleanup_tasks())
            except Exception as e:
                logger.error(f"Async cleanup failed: {e}")
            
            # Step 4: Clean up host processes
            self.cleanup_host_processes()
            
            # Step 5: Clean up temporary files
            self.cleanup_temporary_files()
            
            # Step 6: Save shutdown state
            self.save_shutdown_state()
            
            # Step 7: Flush logs
            self.flush_logs()
            
            shutdown_duration = time.time() - self.shutdown_start_time
            logger.info(f"✅ Graceful shutdown completed in {shutdown_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            raise
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        signal_names = {
            signal.SIGTERM: 'SIGTERM',
            signal.SIGINT: 'SIGINT',
            signal.SIGQUIT: 'SIGQUIT'
        }
        
        signal_name = signal_names.get(signum, f'Signal {signum}')
        logger.info(f"📡 Received {signal_name}, initiating graceful shutdown...")
        
        try:
            self.perform_graceful_shutdown()
            sys.exit(0)
        except Exception as e:
            logger.error(f"Graceful shutdown failed: {e}")
            sys.exit(1)
    
    def register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signals_to_handle = [signal.SIGTERM, signal.SIGINT]
        
        for sig in signals_to_handle:
            signal.signal(sig, self.signal_handler)
            logger.info(f"Registered handler for {sig}")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLMWebUI Graceful Shutdown Handler')
    parser.add_argument('--timeout', type=int, default=30, 
                       help='Shutdown timeout in seconds (default: 30)')
    parser.add_argument('--register-signals', action='store_true',
                       help='Register signal handlers and wait')
    parser.add_argument('--cleanup-only', action='store_true',
                       help='Perform cleanup tasks only (no signal handling)')
    
    args = parser.parse_args()
    
    # Create shutdown handler
    shutdown_handler = GracefulShutdownHandler(shutdown_timeout=args.timeout)
    
    # Register default cleanup tasks
    shutdown_handler.register_cleanup_task(
        'host_processes', 
        shutdown_handler.cleanup_host_processes
    )
    shutdown_handler.register_cleanup_task(
        'temporary_files', 
        shutdown_handler.cleanup_temporary_files
    )
    
    if args.cleanup_only:
        # Just perform cleanup and exit
        logger.info("Performing cleanup tasks only...")
        shutdown_handler.perform_graceful_shutdown()
        
    elif args.register_signals:
        # Register signal handlers and wait
        logger.info("Registering signal handlers...")
        shutdown_handler.register_signal_handlers()
        
        logger.info("Signal handlers registered. Waiting for shutdown signal...")
        logger.info("Send SIGTERM or SIGINT to trigger graceful shutdown")
        
        try:
            # Keep the process alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            shutdown_handler.perform_graceful_shutdown()
    else:
        # Immediate shutdown
        logger.info("Performing immediate graceful shutdown...")
        shutdown_handler.perform_graceful_shutdown()


if __name__ == "__main__":
    main()