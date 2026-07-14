from .common import _collapse_newlines, _collapse_horizontal_spaces, _is_word_char

def fix_streaming_token_spacing(accumulated: str, new_chunk: str) -> str:
    """Intelligently join streaming tokens to avoid spacing issues."""
    if not accumulated:
        return new_chunk

    if not new_chunk:
        return accumulated

    no_space_before = set('.,;:!?%)]}\'"\n')
    no_space_after = set('([{$\n+-')

    last_char = accumulated[-1] if accumulated else ''
    first_char = new_chunk[0] if new_chunk else ''

    needs_space = True

    if last_char in ' \n\t':
        needs_space = False
    elif first_char in ' \n\t':
        needs_space = False
    elif first_char in no_space_before:
        needs_space = False
    elif last_char in no_space_after:
        needs_space = False
    elif last_char.isdigit() and first_char in '.,:%':
        needs_space = False
    elif last_char in '.,' and first_char.isdigit():
        needs_space = False
    elif first_char == "'":
        needs_space = False
    elif last_char in '([{':
        needs_space = False
    elif first_char in ')]}':
        needs_space = False
    elif first_char == '-' and (accumulated.rstrip()[-1:].isalpha() or accumulated.rstrip()[-1:].isdigit()):
        needs_space = False
    elif last_char == '-':
        needs_space = False

    if needs_space:
        return accumulated + ' ' + new_chunk
    return accumulated + new_chunk


def normalize_plain_text_spacing(text: str) -> str:
    """Normalize whitespace for plain text responses without forcing markdown."""
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)

    normalized = text.replace('\r', '')
    
    # Trim horizontal spaces surrounding newlines
    lines = normalized.split('\n')
    normalized = '\n'.join(line.strip(' \t') for line in lines)
    
    normalized = _collapse_newlines(normalized, 2)
    normalized = _collapse_horizontal_spaces(normalized)

    # Tidy spacing around punctuation and digits
    result = []
    for char in normalized:
        if char in ',.;:!?%':
            while result and result[-1].isspace():
                result.pop()
        result.append(char)
    normalized = "".join(result)

    # Brackets spacing cleanup
    result = []
    i = 0
    n = len(normalized)
    while i < n:
        char = normalized[i]
        if char in ('(', '['):
            result.append(char)
            while i + 1 < n and normalized[i + 1].isspace():
                i += 1
        elif char in (')', ']'):
            while result and result[-1].isspace():
                result.pop()
            result.append(char)
        else:
            result.append(char)
        i += 1
    normalized = "".join(result)

    # Remove space between digits
    result = []
    i = 0
    n = len(normalized)
    while i < n:
        char = normalized[i]
        if char.isspace() and i > 0 and i + 1 < n and normalized[i - 1].isdigit():
            j = i
            while j < n and normalized[j].isspace():
                j += 1
            if j < n and normalized[j].isdigit():
                i = j
                result.append(normalized[i])
                i += 1
                continue
        result.append(char)
        i += 1
    normalized = "".join(result)

    # Remove space between digit and percent
    result = []
    i = 0
    n = len(normalized)
    while i < n:
        char = normalized[i]
        if char.isspace() and i > 0 and i + 1 < n and normalized[i - 1].isdigit():
            j = i
            while j < n and normalized[j].isspace():
                j += 1
            if j < n and normalized[j] == '%':
                i = j
                result.append(normalized[i])
                i += 1
                continue
        result.append(char)
        i += 1
    normalized = "".join(result)

    # Hyphen spacing cleanup
    result = []
    i = 0
    n = len(normalized)
    while i < n:
        char = normalized[i]
        if char == '-':
            back_idx = len(result) - 1
            while back_idx >= 0 and result[back_idx].isspace():
                back_idx -= 1
            fwd_idx = i + 1
            while fwd_idx < n and normalized[fwd_idx].isspace():
                fwd_idx += 1
            
            if back_idx >= 0 and _is_word_char(result[back_idx]) and fwd_idx < n and _is_word_char(normalized[fwd_idx]):
                result = result[:back_idx + 1]
                result.append('-')
                i = fwd_idx
                continue
        result.append(char)
        i += 1
    normalized = "".join(result)

    # Contraction spacing cleanup: don' t -> don't, here' s -> here's
    result = []
    i = 0
    n = len(normalized)
    while i < n:
        char = normalized[i]
        if char == "'":
            while result and result[-1].isspace():
                result.pop()
            result.append(char)
            if i + 1 < n and normalized[i + 1].isspace():
                j = i + 1
                while j < n and normalized[j].isspace():
                    j += 1
                if j < n and normalized[j].islower():
                    i = j
                    result.append(normalized[i])
                    i += 1
                    continue
        else:
            result.append(char)
        i += 1
    normalized = "".join(result)

    return normalized.strip()
