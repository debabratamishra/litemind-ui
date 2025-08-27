"""
vLLM service for model management and inference.
Handles Huggingface authentication, model downloading, and streaming responses.

This service supports both native and containerized execution environments:
- Native: Runs vLLM processes directly using asyncio subprocess
- Containerized: Spawns vLLM processes on the host system using conda environments

Key containerized features:
- Automatic detection of container vs native execution
- Host system process management using Docker volume mounts and host network
- Conda environment activation (llm_ui) for vLLM service spawning
- Container-aware process detection and management methods
- OS-independent cache directory handling
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import sys
import shutil
import signal
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Union
import httpx
import psutil
from huggingface_hub import login, list_models, model_info, HfApi
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)

class VLLMService:
    """Service for managing vLLM models and inference."""
    
    def __init__(self):
        self.hf_token = None
        self.current_model = None
        self.server_process = None
        self.host_process_pid = None  # Track host process PID when running in container
        self.containerized_mode = os.path.exists('/.dockerenv')  # Detect if running in container
        self._initialize_config()
    
    def _initialize_config(self):
        """Initialize configuration based on execution environment."""
        try:
            from app.services.host_service_manager import host_service_manager
            self.host_service_manager = host_service_manager
            config = host_service_manager.environment_config
            self.is_containerized = config.is_containerized
            self.api_base = f"{config.vllm_url}/v1"
            self.models_cache_dir = config.hf_cache_dir / "hub"
            logger.info(f"vLLM service configured for {'container' if config.is_containerized else 'native'} environment")
            logger.info(f"API base: {self.api_base}, Cache dir: {self.models_cache_dir}")
        except ImportError:
            # Fallback configuration
            self.is_containerized = False
            self.host_service_manager = None
            self.api_base = "http://localhost:8001/v1"
            self.models_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            logger.warning("Host service manager not available, using default vLLM configuration")
    
    def _check_vllm_available(self) -> bool:
        """Check if vLLM is available in the llm_ui environment"""
        if self.containerized_mode:
            # Running in Docker container - vLLM intentionally not installed
            logger.warning("vLLM not available in containerized mode")
            return False
        
        try:
            # Test if vLLM is available in llm_ui environment
            import subprocess
            result = subprocess.run(
                ["/bin/bash", "-c", "conda activate llm_ui && python -c 'import vllm; print(vllm.__version__)'"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                logger.info(f"vLLM available in llm_ui environment, version: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"vLLM check failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking vLLM availability: {str(e)}")
            return False
    
    def _on_cpu(self) -> bool:
        """Best-effort detection of CPU-only runtime."""
        try:
            import torch  # type: ignore
            has_cuda = getattr(torch, "cuda", None) and torch.cuda.is_available()
            has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            return not bool(has_cuda or has_mps)
        except Exception:
            # If torch isn't importable or any error occurs, assume CPU.
            return True

    def _check_conda_environment(self) -> bool:
        """Check if the vllm_service conda environment exists."""
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return "vllm_service" in result.stdout
            return False
        except Exception as e:
            logger.warning(f"Could not check conda environments: {e}")
            return False

    def _get_conda_activation_command(self) -> str:
        """Get conda activation command for llm_ui environment"""
        if self.containerized_mode:
            return ""
        
        # Use llm_ui environment as requested by user
        return "conda activate llm_ui && "

    def _find_host_vllm_processes(self) -> List[int]:
        """Find existing vLLM processes running on the host system."""
        vllm_pids = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('vllm' in str(arg).lower() for arg in cmdline):
                        # Check if it's a vLLM server process (not just any vllm import)
                        if any('serve' in str(arg) or 'api_server' in str(arg) or '--port' in str(arg) for arg in cmdline):
                            vllm_pids.append(proc.info['pid'])
                            logger.info(f"Found vLLM process: PID {proc.info['pid']}, CMD: {' '.join(cmdline)}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            logger.error(f"Error finding vLLM processes: {e}")
        
        return vllm_pids

    def _kill_host_process(self, pid: int) -> bool:
        """Kill a process on the host system by PID."""
        try:
            if self.is_containerized:
                # When in container, we need to kill the process on the host
                # This requires the container to have access to host processes
                # We'll use a system call that can reach the host process
                result = subprocess.run(
                    ["kill", "-TERM", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"Successfully sent TERM signal to host process {pid}")
                    return True
                else:
                    # Try SIGKILL if SIGTERM fails
                    result = subprocess.run(
                        ["kill", "-KILL", str(pid)],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        logger.info(f"Successfully sent KILL signal to host process {pid}")
                        return True
                    else:
                        logger.error(f"Failed to kill host process {pid}: {result.stderr}")
                        return False
            else:
                # Native execution - use psutil
                proc = psutil.Process(pid)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                logger.info(f"Successfully killed native process {pid}")
                return True
        except Exception as e:
            logger.error(f"Error killing process {pid}: {e}")
            return False

    async def _check_host_process_running(self, pid: int) -> bool:
        """Check if a process is running on the host system."""
        try:
            if self.is_containerized:
                # Check if process exists on host system
                result = subprocess.run(
                    ["ps", "-p", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0
            else:
                # Native execution - use psutil
                return psutil.pid_exists(pid)
        except Exception as e:
            logger.error(f"Error checking process {pid}: {e}")
            return False

    def _scrub_hf_env(self, env: Dict[str, str]) -> Dict[str, str]:
        """Remove HF auth variables so downstream libs don't send bad Authorization headers."""
        for k in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            env.pop(k, None)
        return env

    def _build_vllm_cmd(self, model_name: str, **kwargs) -> List[str]:
        """Build the command to start the vLLM OpenAI-compatible server.
        Handles both containerized and native execution environments."""
        
        # Get conda activation command if needed
        conda_cmd = self._get_conda_activation_command()
        
        # Build the base vLLM command
        vllm_cli = shutil.which("vllm")
        if vllm_cli:
            # Newer vLLM exposes `vllm serve <model>`
            base_cmd = [vllm_cli, "serve", model_name]
            # host/port flags are supported by `vllm serve`
            base_cmd.extend(["--host", "0.0.0.0", "--port", "8001"])
        else:
            # Fallback to python -m entrypoint
            if self.is_containerized:
                # In container, use python from the conda environment on host
                base_cmd = ["python", "-m", "vllm.entrypoints.openai.api_server",
                           "--host", "0.0.0.0", "--port", "8001", "--model", model_name]
            else:
                # Native execution - use current interpreter
                base_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                           "--host", "0.0.0.0", "--port", "8001", "--model", model_name]

        # Dtype: default to 'auto' to work on CPU/GPU without forcing half
        requested_dtype = kwargs.get("dtype")
        if requested_dtype is None:
            dtype = "auto"
        else:
            dtype = str(requested_dtype)
            if dtype.lower() in {"half", "fp16", "float16"} and self._on_cpu():
                logger.info("Overriding dtype 'half' to 'auto' on CPU-only runtime.")
                dtype = "auto"
        base_cmd.extend(["--dtype", dtype])

        # Optional arguments (only include if explicitly provided). On CPU, skip GPU memory utilization.
        if "max_model_len" in kwargs and kwargs["max_model_len"] is not None:
            base_cmd.extend(["--max-model-len", str(kwargs["max_model_len"])])
        if ("gpu_memory_utilization" in kwargs and kwargs["gpu_memory_utilization"] is not None
                and not self._on_cpu()):
            base_cmd.extend(["--gpu-memory-utilization", str(kwargs["gpu_memory_utilization"])])

        # Combine conda activation with vLLM command
        if conda_cmd:
            # Use conda run to execute in the llm_ui environment
            full_cmd = conda_cmd + base_cmd
        else:
            full_cmd = base_cmd

        logger.info(f"Built vLLM command: {' '.join(full_cmd)}")
        return full_cmd
        
    async def _is_process_running(self, process: Optional[Union[asyncio.subprocess.Process, int]]) -> bool:
        """Check if a process is still running (handles both asyncio subprocess and PID)."""
        if process is None:
            return False
        
        try:
            if isinstance(process, int):
                # It's a PID - check if host process is running
                return await self._check_host_process_running(process)
            else:
                # It's an asyncio subprocess
                if hasattr(process, 'returncode') and process.returncode is not None:
                    return False
                # If returncode is None, process is still running
                return True
        except Exception:
            # Fallback: try to check if we can get status without blocking
            try:
                if isinstance(process, int):
                    return await self._check_host_process_running(process)
                else:
                    await asyncio.wait_for(process.wait(), timeout=0.001)
                    return False  # Process completed
            except asyncio.TimeoutError:
                return True  # Still running
            except Exception:
                return False  # Assume not running if error

    async def _server_model_ids(self) -> List[str]:
        """Return the list of model ids the vLLM server thinks it serves."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.api_base}/models")
                r.raise_for_status()
                payload = r.json() or {}
                return [m.get("id") for m in payload.get("data", []) if isinstance(m, dict) and m.get("id")]
        except Exception:
            return []

    def _is_model_public(self, repo_id: str) -> Optional[bool]:
        """Return True if the model repo is public, False if private/gated, None if unknown."""
        try:
            api = HfApi()
            api.model_info(repo_id, token=None)
            return True
        except HfHubHTTPError as e:
            try:
                code = getattr(e.response, "status_code", None)
            except Exception:
                code = None
            if code in (401, 403):
                return False
            if code == 404:
                # Could be missing or private; treat as unknown so caller can decide.
                return None
            return None
        except Exception:
            return None
        
    def set_hf_token(self, token: str) -> Dict[str, str]:
        """Set and validate Huggingface token.
        If a model is already loaded/served, skip validation since inference
        on local/cached models does not require an online login.
        """
        # If a model is already loaded by the running vLLM server, the token is not
        # required for inference. In that case, avoid calling `login()` to prevent
        # spurious warnings and failures when users don't have/need a token.
        if self.current_model:
            env_token = os.environ.get("HF_TOKEN")
            if token and token == self.hf_token:
                logger.info("Huggingface token already set; skipping revalidation (model already loaded)")
                return {"status": "success", "message": "Token already set"}
            if token and env_token and token == env_token:
                logger.info("HF_TOKEN already present in environment; skipping revalidation (model already loaded)")
                self.hf_token = token
                return {"status": "success", "message": "Token already set in environment"}
            if token:
                # Store for future downloads, but do not validate/login now.
                self.hf_token = token
                os.environ["HF_TOKEN"] = token
                logger.info("Model already loaded; storing HF token without validation")
                return {"status": "success", "message": "Token stored; validation skipped for loaded model"}
            else:
                logger.info("Model already loaded; token not required for inference")
                return {"status": "success", "message": "Token not required for loaded local model"}

        # If no model is loaded, proceed with normal validation logic (used for downloads
        # and for starting models that require Hub access).
        if token and token == self.hf_token:
            logger.info("Huggingface token already set; skipping revalidation")
            return {"status": "success", "message": "Token already set"}
        try:
            # If an env token is already set and matches, prefer that and skip login
            env_token = os.environ.get("HF_TOKEN")
            if env_token and (not token or token == env_token):
                self.hf_token = env_token
                logger.info("Using existing HF_TOKEN from environment; skipping explicit login")
                return {"status": "success", "message": "Using existing environment token"}

            # Validate token by attempting login
            if not token:
                return {"status": "error", "message": "No token provided and no model loaded; token required for Hub access"}

            login(token=token, add_to_git_credential=True)
            self.hf_token = token
            os.environ["HF_TOKEN"] = token
            logger.info("Huggingface token validated and set successfully")
            return {"status": "success", "message": "Token validated successfully"}
        except Exception as e:
            logger.error(f"Failed to validate HF token: {e}")
            return {"status": "error", "message": f"Invalid token: {str(e)}"}
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of locally cached models and popular models from HF."""
        local_models = []
        popular_models = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large", 
            "google/gemma-2b-it",
            "google/gemma-7b-it",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "teknium/OpenHermes-2.5-Mistral-7B",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ]
        
        # Check for locally cached models
        if self.models_cache_dir.exists():
            for model_dir in self.models_cache_dir.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                    # Convert cache dir name back to model name
                    model_name = model_dir.name.replace('--', '/')
                    local_models.append(model_name)
        
        return {
            "local_models": local_models,
            "popular_models": popular_models
        }
    
    async def download_model(self, model_name: str) -> Dict[str, str]:
        """Download a model from Huggingface Hub.
        Supports public models without requiring a token.
        """
        try:
            # Detect whether repo is public when no token is available
            token = self.hf_token
            if not token:
                public = self._is_model_public(model_name)
                if public is False:
                    return {"status": "error", "message": f"Model {model_name} requires authentication. Please set a Hugging Face token."}
                # If public (True) or unknown (None), attempt download without token

            # Prepare environment: include token only if we have a validated one
            dl_env = os.environ.copy()
            if token:
                dl_env["HF_TOKEN"] = token
            else:
                self._scrub_hf_env(dl_env)

            # Verify access (optional but yields nicer errors)
            api = HfApi(token=token)
            try:
                _ = api.model_info(model_name)
                logger.info(f"Model {model_name} found on HF Hub")
            except HfHubHTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    return {"status": "error", "message": f"Model {model_name} not found"}
                if e.response is not None and e.response.status_code == 403:
                    return {"status": "error", "message": f"Access denied to {model_name}. Check token permissions."}
                raise

            # Download model (this will cache it locally)
            logger.info(f"Starting download of model: {model_name}")
            cmd = [
                "python", "-c",
                (
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download('{model_name}'); "
                    "print('Download completed')"
                ),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=dl_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                logger.info(f"Successfully downloaded model: {model_name}")
                return {"status": "success", "message": f"Model {model_name} downloaded successfully"}
            else:
                error_msg = stderr.decode() if stderr else "Unknown error during download"
                logger.error(f"Failed to download model: {error_msg}")
                return {"status": "error", "message": f"Download failed: {error_msg}"}

        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def start_vllm_server(self, model_name: str, **kwargs) -> Dict[str, str]:
        """Start vLLM server with specified model."""
        try:
            # Ensure vLLM is importable in this environment
            missing_msg = self._check_vllm_available()
            if missing_msg:
                logger.error(missing_msg)
                return {"status": "error", "message": missing_msg}
            # Stop existing server if running
            if self.server_process and await self._is_process_running(self.server_process):
                logger.info("Stopping existing vLLM server...")
                await self.stop_vllm_server()
                # Wait a bit for cleanup
                await asyncio.sleep(2)

            # Normalize model name: if a non-existent local path like 'models/...',
            # treat it as a Hugging Face repo ID. If a local path exists but lacks
            # config.json, return a helpful error.
            original_model_name = model_name
            local_dir = Path(model_name)
            if local_dir.exists():
                if not (local_dir / "config.json").exists():
                    msg = (
                        f"Provided local model path '{model_name}' exists but is missing 'config.json'. "
                        "Point to the directory that contains a valid Hugging Face model (with config.json), "
                        "or use a repo id like 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'."
                    )
                    logger.error(msg)
                    return {"status": "error", "message": msg}
            else:
                if model_name.startswith("models/"):
                    stripped = model_name.split("models/", 1)[1]
                    logger.info(
                        f"Normalizing model name from '{model_name}' to '{stripped}' (treating as HF repo id)."
                    )
                    model_name = stripped

            # Prepare command
            cmd = self._build_vllm_cmd(model_name, **kwargs)

            # Set environment variables with correct auth behavior
            env = os.environ.copy()
            # Decide if we need Hub access (repo id vs local path)
            needs_hub = not Path(model_name).exists()
            if needs_hub:
                public = self._is_model_public(model_name)
                if public is True:
                    # Ensure no Authorization header is sent; invalid env tokens cause 401
                    env = self._scrub_hf_env(env)
                    logger.info("Public repo detected; scrubbing HF auth from server environment")
                elif public is False:
                    if not self.hf_token:
                        msg = ("Model requires Hugging Face authentication. "
                               "Please set a valid token before starting the server.")
                        logger.error(msg)
                        return {"status": "error", "message": msg}
                    env["HF_TOKEN"] = self.hf_token
                else:
                    # Unknown; be conservative. If token is set, include it; otherwise scrub.
                    if self.hf_token:
                        env["HF_TOKEN"] = self.hf_token
                    else:
                        env = self._scrub_hf_env(env)
            else:
                # Local path model—no auth required; scrub to avoid spurious 401s
                env = self._scrub_hf_env(env)

            logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")

            # Start server process - handle containerized vs native execution
            if self.is_containerized:
                # In container: spawn process on host system using the conda environment
                # We need to construct a command that will run on the host system
                try:
                    logger.info(f"Starting vLLM server on host system using conda environment 'llm_ui'")
                    logger.info(f"Command to execute: {' '.join(cmd)}")
                    
                    # Start the process and let it run independently on the host
                    # We use Popen with proper environment setup
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=subprocess.DEVNULL,  # Don't capture output to avoid blocking
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,    # Start in new session to avoid signal propagation
                        cwd=os.path.expanduser("~")  # Set working directory to home
                    )
                    
                    self.host_process_pid = process.pid
                    self.server_process = process.pid  # Store PID for tracking
                    
                    logger.info(f"Started vLLM server on host system with PID: {process.pid}")
                    
                    # Don't wait for the process - let it run independently
                    # The process will continue running on the host even if container restarts
                    
                except Exception as e:
                    logger.error(f"Failed to start vLLM process on host: {e}")
                    return {"status": "error", "message": f"Failed to start host process: {str(e)}"}
            else:
                # Native execution: use asyncio subprocess as before
                self.server_process = await asyncio.create_subprocess_exec(
                    *cmd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                logger.info(f"Started vLLM server natively with asyncio subprocess")

            # Wait a bit for server to start
            logger.info("Waiting for vLLM server to start...")
            await asyncio.sleep(10)  # Increased wait time

            # Check if server is running and responsive
            max_retries = 30  # 30 seconds total wait
            for attempt in range(max_retries):
                if await self.is_server_running():
                    # Prefer the exact id(s) the server reports, fall back to our name.
                    try:
                        ids = await self._server_model_ids()
                        if ids:
                            preferred = None
                            for cand in (model_name, original_model_name):
                                if cand in ids:
                                    preferred = cand
                                    break
                            self.current_model = preferred or ids[0]
                        else:
                            self.current_model = model_name
                    except Exception:
                        self.current_model = model_name
                    logger.info(f"vLLM server started successfully with model id: {self.current_model}")
                    return {"status": "success", "message": f"Server started with {self.current_model}"}

                # Check if process died
                if not await self._is_process_running(self.server_process):
                    # Get error output - handle both process types
                    try:
                        if self.is_containerized and isinstance(self.server_process, int):
                            # For host processes, we can't easily get stdout/stderr
                            # But we can check if there are any vLLM processes running
                            running_vllm_pids = self._find_host_vllm_processes()
                            if running_vllm_pids:
                                logger.info(f"Found running vLLM processes: {running_vllm_pids}")
                                # Update our tracked process to one of the running ones
                                self.server_process = running_vllm_pids[0]
                                continue  # Go back and check if server is responsive
                            
                            error_msg = (
                                f"vLLM server process (PID: {self.server_process}) terminated during startup.\n"
                                f"Command: {' '.join(cmd)}\n"
                                "Check that:\n"
                                "1. vLLM is installed in the 'vllm_service' conda environment on the host\n"
                                "2. The model name is correct\n"
                                "3. There's sufficient memory available\n"
                                "Run this command on the host to test: conda activate vllm_service && vllm serve --help"
                            )
                        else:
                            # For asyncio subprocess, get output as before
                            stdout, stderr = await self.server_process.communicate()
                            stderr_txt = (stderr or b"").decode(errors="ignore")
                            stdout_txt = (stdout or b"").decode(errors="ignore")
                            error_msg = (
                                "vLLM server process terminated during startup.\n"
                                f"Command: {' '.join(cmd)}\n"
                                f"STDERR (tail):\n{stderr_txt[-4000:]}\n"
                                f"STDOUT (tail):\n{stdout_txt[-4000:]}"
                            )
                        logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
                    except Exception as e:
                        logger.error(f"Failed to get process output: {e}")
                        return {"status": "error", "message": "Server process failed"}

                await asyncio.sleep(1)

            # If we get here, server didn't become ready in time
            logger.error("vLLM server failed to become ready in time")
            await self.stop_vllm_server()
            return {"status": "error", "message": "Server failed to start - timeout waiting for readiness"}

        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            return {"status": "error", "message": str(e)}
    
    async def stop_vllm_server(self) -> Dict[str, str]:
        """Stop the vLLM server (handles both containerized and native processes)."""
        try:
            if self.server_process and await self._is_process_running(self.server_process):
                logger.info("Terminating vLLM server...")
                
                if self.is_containerized and isinstance(self.server_process, int):
                    # Handle host process termination
                    success = self._kill_host_process(self.server_process)
                    if success:
                        # Wait a bit for process to terminate
                        await asyncio.sleep(2)
                        # Verify it's stopped
                        if not await self._check_host_process_running(self.server_process):
                            logger.info(f"vLLM host process {self.server_process} stopped successfully")
                        else:
                            logger.warning(f"vLLM host process {self.server_process} may still be running")
                    else:
                        logger.error(f"Failed to stop vLLM host process {self.server_process}")
                        return {"status": "error", "message": f"Failed to stop host process {self.server_process}"}
                else:
                    # Handle asyncio subprocess termination
                    self.server_process.terminate()
                    
                    try:
                        # Wait for graceful shutdown
                        await asyncio.wait_for(self.server_process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Force kill if it doesn't shutdown gracefully
                        logger.warning("vLLM server didn't shutdown gracefully, forcing kill...")
                        self.server_process.kill()
                        await self.server_process.wait()
                
                self.server_process = None
                self.host_process_pid = None
                self.current_model = None
                logger.info("vLLM server stopped")
                return {"status": "success", "message": "Server stopped"}
            else:
                # Check for any orphaned vLLM processes and clean them up
                if self.is_containerized:
                    orphaned_pids = self._find_host_vllm_processes()
                    if orphaned_pids:
                        logger.info(f"Found orphaned vLLM processes: {orphaned_pids}")
                        for pid in orphaned_pids:
                            self._kill_host_process(pid)
                        return {"status": "success", "message": f"Cleaned up {len(orphaned_pids)} orphaned processes"}
                
                return {"status": "info", "message": "No server running"}
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            self.server_process = None  # Reset even on error
            self.host_process_pid = None
            self.current_model = None
            return {"status": "error", "message": str(e)}
    
    async def is_server_running(self) -> bool:
        """Check if vLLM server is running and responsive."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.api_base}/models")
                return response.status_code == 200
        except Exception:
            return False

    async def get_server_status(self) -> Dict[str, any]:
        """Get comprehensive server status including process information."""
        status = {
            "is_running": False,
            "is_responsive": False,
            "process_info": None,
            "host_processes": [],
            "current_model": self.current_model,
            "execution_mode": "containerized" if self.is_containerized else "native"
        }
        
        # Check if server is responsive
        status["is_responsive"] = await self.is_server_running()
        
        # Check process status
        if self.server_process:
            status["is_running"] = await self._is_process_running(self.server_process)
            if self.is_containerized and isinstance(self.server_process, int):
                status["process_info"] = {
                    "type": "host_process",
                    "pid": self.server_process,
                    "running": status["is_running"]
                }
            else:
                status["process_info"] = {
                    "type": "asyncio_subprocess",
                    "running": status["is_running"]
                }
        
        # Find all vLLM processes on host (useful for debugging)
        if self.is_containerized:
            host_pids = self._find_host_vllm_processes()
            status["host_processes"] = host_pids
        
        return status

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Naive messages->prompt conversion for fallback to /v1/completions."""
        parts: List[str] = []
        for m in messages:
            role = (m.get("role") or "user").strip()
            content = (m.get("content") or "").strip()
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    async def stream_vllm_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from vLLM server."""

        if not await self.is_server_running():
            logger.error("vLLM server is not running")
            yield "Error: vLLM server is not running"
            return

        try:
            # Align model id with what the server actually serves
            server_ids = await self._server_model_ids()
            model_to_use = (model or self.current_model)
            if server_ids and model_to_use not in server_ids:
                logger.warning(
                    f"Requested model '{model_to_use}' not in server ids {server_ids}; using '{server_ids[0]}' instead."
                )
                model_to_use = server_ids[0]

            url_chat = f"{self.api_base}/chat/completions"
            payload_chat = {
                "model": model_to_use,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            logger.info(f"Sending request to vLLM at: {url_chat}")
            logger.info(f"Payload: {payload_chat}")

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", url_chat, json=payload_chat) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 404:
                    # Fallback to the plain completions API if chat route is unavailable
                    prompt = self._messages_to_prompt(messages)
                    url_cmp = f"{self.api_base}/completions"
                    payload_cmp = {
                        "model": model_to_use,
                        "prompt": prompt,
                        "stream": True,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    logger.warning("/chat/completions returned 404. Falling back to /completions.")
                    logger.info(f"Sending request to vLLM at: {url_cmp}")
                    logger.info(f"Payload: {payload_cmp}")

                    async with httpx.AsyncClient(timeout=120.0) as client:
                        async with client.stream("POST", url_cmp, json=payload_cmp) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line or not line.startswith("data: "):
                                    continue
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    text = data.get("choices", [{}])[0].get("text", "")
                                    if text:
                                        yield text
                                except json.JSONDecodeError:
                                    continue
                else:
                    raise
        except Exception as e:
            error_msg = f"vLLM streaming error: {str(e)}"
            logger.error(error_msg)
            yield f"Error: {error_msg}"

# Global vLLM service instance
vllm_service = VLLMService()
