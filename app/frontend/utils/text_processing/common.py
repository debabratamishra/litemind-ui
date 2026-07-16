from typing import Set


def _load_common_words() -> Set[str]:
    manual = {
        'temperatures', 'information', 'conditions', 'performance', 'requirements',
        'available', 'different', 'important', 'following', 'including',
        'experience', 'development', 'government', 'environment', 'management',
        'international', 'community', 'university', 'organization', 'relationship',
        'opportunity', 'significant', 'particularly', 'understanding', 'responsibilities',
    }

    try:
        from wordfreq import top_n_list

        words = set(w.lower() for w in top_n_list("en", 20000))
        long_words = {w for w in words if len(w) >= 6}
        return long_words.union(manual)
    except Exception:
        return manual


COMMON_WORDS = _load_common_words()


def _collapse_newlines(text: str, max_count: int = 2) -> str:
    """Collapses consecutive newlines to max_count in O(n) time."""
    result = []
    newline_count = 0
    for char in text:
        if char == '\n':
            newline_count += 1
            if newline_count <= max_count:
                result.append(char)
        else:
            newline_count = 0
            result.append(char)
    return "".join(result)


def _collapse_horizontal_spaces(text: str) -> str:
    """Collapse consecutive spaces/tabs to a single space in O(n) time."""
    result = []
    space_count = 0
    for char in text:
        if char in (' ', '\t'):
            space_count += 1
            if space_count == 1:
                result.append(' ')
        else:
            space_count = 0
            result.append(char)
    return "".join(result)


def _is_word_char(c: str) -> bool:
    """Check if character is alphanumeric or underscore (equivalent to \\w)."""
    return c.isalnum() or c == '_'
