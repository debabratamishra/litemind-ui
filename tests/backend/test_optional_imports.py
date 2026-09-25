"""The FastAPI app must import without the optional ``voice`` dependency group.

``pipecat-ai`` ships in the optional ``voice`` group. ``backend.main`` mounts
the voice router at import time, so a module-level Pipecat import makes the
whole backend unstartable for anyone who installed the default groups.
"""

import subprocess
import sys
import textwrap

_BLOCK_PIECAT = textwrap.dedent(
    """
    import sys


    class _BlockPipecat:
        def find_spec(self, name, path=None, target=None):
            if name == "pipecat" or name.startswith("pipecat."):
                raise ImportError("pipecat is not installed (simulated)")
            return None


    sys.meta_path.insert(0, _BlockPipecat())
    for _name in [m for m in sys.modules if m == "backend.main" or m.startswith("pipecat")]:
        del sys.modules[_name]

    import backend.main

    assert backend.main.app is not None
    print("IMPORT_OK")
    """
)


def test_backend_main_imports_without_pipecat():
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_PIECAT],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout
