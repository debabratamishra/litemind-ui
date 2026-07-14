import json

def unescape_text(s: str) -> str:
    """Convert visible escape sequences into their actual characters."""
    if not isinstance(s, str):
        s = str(s)

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    try:
        s = json.loads(f'"{s}"')
    except Exception:
        s = s.replace('\\n', '\n').replace('\\t', '\t')

    return s.replace('\\r\\n', '\n').replace('\r\n', '\n')
