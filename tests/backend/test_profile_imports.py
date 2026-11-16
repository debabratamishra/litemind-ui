"""Heavy imports must stay out of module bodies so a core-only image boots.

The uv dependency groups split the install into ``core``/``ml``/``voice``/
``ocr`` profiles. The heavy packages below ship only in the ml/voice/ocr
groups, so no module in ``backend/app`` (or ``backend/main.py``) may bind
them at import time; they are imported at call time instead. ``backend.main``
additionally must import ``RAGService`` and the speech preloads only inside
functions (the lifespan), never at module level.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "backend" / "main.py"
APP = ROOT / "backend" / "app"

# Import roots that ship in the ml/voice/ocr profiles, never in core.
HEAVY = {
    "chromadb",
    "torch",
    "transformers",
    "pipecat",
    "librosa",
    "easyocr",
    "cv2",
    "crewai",
}

# Names backend/main.py may only import inside functions.
LAZY_IN_MAIN = {"RAGService", "preload_stt_model", "preload_tts_model"}


def _module_body_imports(tree):
    """(root, node) for every import the module executes at import time."""
    found = []
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            continue
        if isinstance(node, ast.Import):
            found.extend((alias.name.split(".")[0], node) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.extend((node.module.split(".")[0], node) for _ in node.names)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _in_function(tree, node):
    """True when node sits inside a FunctionDef/AsyncFunctionDef body."""
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        cur = parents.get(cur)
    return False


def _bound_names(node):
    if isinstance(node, ast.ImportFrom):
        return [alias.asname or alias.name for alias in node.names]
    return [alias.asname or alias.name.split(".")[0] for alias in node.names]


def test_no_heavy_module_body_imports():
    offenders = []
    for path in [MAIN] + sorted(APP.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for root, node in _module_body_imports(tree):
            if root in HEAVY:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {root}")
    assert not offenders, (
        "heavy imports must move inside functions or TYPE_CHECKING:\n" + "\n".join(offenders)
    )


def test_main_imports_rag_and_speech_preloads_only_inside_functions():
    tree = ast.parse(MAIN.read_text())
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        bound = set(_bound_names(node)) & LAZY_IN_MAIN
        if bound:
            assert _in_function(tree, node), f"module-body import of {sorted(bound)}"
            seen |= bound
    assert seen == LAZY_IN_MAIN
