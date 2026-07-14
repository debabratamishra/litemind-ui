from .common import COMMON_WORDS, _collapse_newlines, _collapse_horizontal_spaces, _is_word_char

def _fix_newlines_between_letters(text: str) -> str:
    """Join characters/letters separated by newlines during streaming."""
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '\n' and i > 0 and i + 1 < n and text[i - 1].isalpha() and text[i + 1].isalpha():
            after_letter2 = text[i + 2] if i + 2 < n else ''
            if not after_letter2 or after_letter2 == '\n' or not after_letter2.isalpha():
                i += 1
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _join_spaced_letters(text: str) -> str:
    """Join single letters separated by spaces back into words."""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        if _is_word_char(text[i]):
            start = i
            while i < n and _is_word_char(text[i]):
                i += 1
            tokens.append((True, text[start:i]))
        elif text[i].isspace():
            start = i
            while i < n and text[i].isspace():
                i += 1
            tokens.append((False, text[start:i]))
        else:
            tokens.append((False, text[i]))
            i += 1
    
    result_tokens = []
    t_idx = 0
    num_tokens = len(tokens)
    while t_idx < num_tokens:
        run = []
        temp_idx = t_idx
        while temp_idx < num_tokens:
            is_word, val = tokens[temp_idx]
            if is_word and len(val) == 1 and val.isalpha():
                run.append(temp_idx)
                if temp_idx + 2 < num_tokens:
                    next_is_space, space_val = tokens[temp_idx + 1]
                    next_is_word, word_val = tokens[temp_idx + 2]
                    if not next_is_space and space_val.isspace() and next_is_word and len(word_val) == 1 and word_val.isalpha():
                        temp_idx += 2
                        continue
                break
            else:
                break
        
        if len(run) >= 4:
            merged_word = "".join(tokens[idx][1] for idx in run)
            if t_idx >= 2:
                prev_space_ok, prev_space_val = tokens[t_idx - 1]
                prev_word_ok, prev_word_val = tokens[t_idx - 2]
                if not prev_space_ok and prev_space_val.isspace() and prev_word_ok and len(prev_word_val) >= 2:
                    result_tokens.pop()  # pop space
                    prev_word = result_tokens.pop()[1]  # pop word
                    merged_word = prev_word + merged_word
            
            result_tokens.append((True, merged_word))
            t_idx = run[-1] + 1
            continue
        
        if t_idx + 2 < num_tokens:
            w1_ok, w1_val = tokens[t_idx]
            sp_ok, sp_val = tokens[t_idx + 1]
            w2_ok, w2_val = tokens[t_idx + 2]
            if w1_ok and w1_val.isupper() and 1 <= len(w1_val) <= 2:
                if not sp_ok and sp_val.isspace() and ' ' in sp_val:
                    if w2_ok and w2_val.isupper() and len(w2_val) == 1:
                        result_tokens.append((True, w1_val + w2_val))
                        t_idx += 3
                        continue
        
        result_tokens.append(tokens[t_idx])
        t_idx += 1
        
    return "".join(val for _, val in result_tokens)


def _clean_link_artifacts(s: str) -> str:
    """Manually clean duplicate/broken markdown link artifacts without regex."""
    result = []
    i = 0
    n = len(s)
    while i < n:
        if s[i:].startswith("[Link]("):
            j = i + 7
            while j < n and s[j].isspace():
                j += 1
            if j < n and s[j] == '(':
                result.append('(')
                i = j + 1
                continue
        result.append(s[i])
        i += 1
    s = "".join(result)
    
    result = []
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j < n and s[j].isspace():
            j += 1
        if j < n and s[j] == '-':
            k = j + 1
            while k < n and s[k].isspace():
                k += 1
            if s[k:].startswith("[Link]("):
                p = k + 7
                open_brackets = 1
                while p < n and open_brackets > 0:
                    if s[p] == '(':
                        open_brackets += 1
                    elif s[p] == ')':
                        open_brackets -= 1
                    p += 1
                if open_brackets == 0:
                    q = p
                    while q < n and s[q].isspace():
                        q += 1
                    if q < n and s[q] == '-':
                        r = q + 1
                        while r < n and s[r].isspace():
                            r += 1
                        if s[r:].startswith("[Link]("):
                            t = r + 7
                            open_brackets = 1
                            while t < n and open_brackets > 0:
                                if s[t] == '(':
                                    open_brackets += 1
                                elif s[t] == ')':
                                    open_brackets -= 1
                                t += 1
                            if open_brackets == 0:
                                u = t
                                while u < n and s[u].isspace():
                                    u += 1
                                if u < n and s[u] == '-':
                                    v = u + 1
                                    while v < n and s[v].isspace():
                                        v += 1
                                    if s[v:].lower().startswith("link"):
                                        result.append(" - Link ")
                                        i = v + 4
                                        continue
        result.append(s[i])
        i += 1
    s = "".join(result)
    
    if s.endswith('[Link]('):
        s = s[:-7]
    else:
        temp = s.rstrip()
        if temp.endswith('[Link]('):
            s = temp[:-7]
            
    return s


def format_web_search_response(text: str) -> str:
    """Format web search response text for proper display."""
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)

    text = text.replace('\r', '')

    text = _fix_newlines_between_letters(text)
    text = _join_spaced_letters(text)

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "'" and i > 0 and _is_word_char(text[i - 1]):
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].islower():
                result.append("'")
                result.append(text[j])
                i = j + 1
                continue
        result.append(text[i])
        i += 1
    text = "".join(result)

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isdigit():
            start = i
            while i < n and text[i].isdigit():
                i += 1
            digits = text[start:i]
            j = i
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j] == '°':
                k = j + 1
                while k < n and text[k].isspace():
                    k += 1
                if k < n and text[k] in 'CFcf':
                    result.append(digits + '°' + text[k])
                    i = k + 1
                else:
                    result.append(digits + '°')
                    i = j + 1
                continue
            else:
                result.append(digits)
                continue
        result.append(text[i])
        i += 1
    text = "".join(result)

    tokens = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isalpha():
            start = i
            while i < n and text[i].isalpha():
                i += 1
            tokens.append((True, text[start:i]))
        elif text[i].isspace():
            start = i
            while i < n and text[i].isspace():
                i += 1
            tokens.append((False, text[start:i]))
        else:
            tokens.append((False, text[i]))
            i += 1

    result_tokens = []
    t_idx = 0
    num_tokens = len(tokens)
    while t_idx < num_tokens:
        if t_idx + 2 < num_tokens:
            w1_ok, w1_val = tokens[t_idx]
            sp_ok, sp_val = tokens[t_idx + 1]
            w2_ok, w2_val = tokens[t_idx + 2]
            if w1_ok and w1_val and w1_val[0].isupper() and w1_val[1:].islower():
                if not sp_ok and sp_val.isspace() and ' ' in sp_val:
                    if w2_ok and w2_val.islower() and len(w2_val) >= 3:
                        combined = (w1_val + w2_val).lower()
                        if combined in COMMON_WORDS:
                            result_tokens.append((True, w1_val + w2_val))
                            t_idx += 3
                            continue
        result_tokens.append(tokens[t_idx])
        t_idx += 1
    text = "".join(val for _, val in result_tokens)

    for prefix, full_word in [("s", "south"), ("n", "north"), ("e", "east"), ("w", "west")]:
        start_idx = 0
        while True:
            pos = text.lower().find(prefix + " ", start_idx)
            if pos == -1:
                break
            if pos > 0 and _is_word_char(text[pos - 1]):
                start_idx = pos + 1
                continue
            j = pos + len(prefix)
            while j < len(text) and text[j].isspace():
                j += 1
            if text[j:].lower().startswith(full_word):
                matched_full_word = text[j:j+len(full_word)]
                text = text[:pos] + matched_full_word + text[j+len(full_word):]
                start_idx = pos + len(matched_full_word)
            else:
                start_idx = pos + 1

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isupper():
            start = i
            while i < n and text[i].isupper():
                i += 1
            uppercase_seq = text[start:i]
            if len(uppercase_seq) >= 2 and i < n and text[i] == ':':
                j = i + 1
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j].isupper():
                    u1 = text[j]
                    k = j + 1
                    while k < n and text[k].isspace():
                        k += 1
                    if k < n and text[k].isupper():
                        u2 = text[k]
                        m = k + 1
                        u3 = ""
                        if k > j + 1:
                            while m < n and text[m].isspace():
                                m += 1
                            if m < n and text[m].isupper():
                                u3 = text[m]
                                result.append(uppercase_seq + ":" + u1 + u2 + u3)
                                i = m + 1
                                continue
                            else:
                                result.append(uppercase_seq + ":" + u1 + u2)
                                i = k + 1
                                continue
            result.append(uppercase_seq)
            continue
        result.append(text[i])
        i += 1
    text = "".join(result)

    result = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char.isspace() and i > 0 and i + 1 < n and text[i - 1].isdigit():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].isdigit():
                i = j
                result.append(text[i])
                i += 1
                continue
        result.append(char)
        i += 1
    text = "".join(result)

    result = []
    for char in text:
        if char in ',.;:!?%':
            while result and result[-1].isspace():
                result.pop()
        result.append(char)
    text = "".join(result)

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '$':
            result.append('$')
            i += 1
            while i < n and text[i].isspace():
                i += 1
            continue
        elif text[i:].startswith('Rs'):
            j = i + 2
            if j < n and text[j].isspace():
                result.append('Rs ')
                while j < n and text[j].isspace():
                    j += 1
                i = j
                continue
        result.append(text[i])
        i += 1
    text = "".join(result)

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] in '.,:' and i > 0 and text[i - 1].isdigit():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].isdigit():
                result.append(text[i])
                result.append(text[j])
                i = j + 1
                continue
        result.append(text[i])
        i += 1
    text = "".join(result)

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '-':
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].isdigit():
                result.append('-')
                result.append(text[j])
                i = j + 1
                continue
        elif text[i] == '%':
            back_idx = len(result) - 1
            while back_idx >= 0 and result[back_idx].isspace():
                back_idx -= 1
            if back_idx >= 0 and result[back_idx].isdigit():
                result = result[:back_idx + 1]
            result.append('%')
            i += 1
            continue
        result.append(text[i])
        i += 1
    text = "".join(result)

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '-' and i > 0 and _is_word_char(text[i - 1]):
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and _is_word_char(text[j]):
                result.append('-')
                result.append(text[j])
                i = j + 1
                continue
        result.append(text[i])
        i += 1
    text = "".join(result)

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '+':
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].isdigit():
                result.append('+')
                result.append(text[j])
                i = j + 1
                continue
        result.append(text[i])
        i += 1
    text = "".join(result)

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '[':
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].isdigit():
                digit_start = j
                while j < n and text[j].isdigit():
                    j += 1
                digits = text[digit_start:j]
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j] == ']':
                    result.append('[' + digits + ']')
                    i = j + 1
                    continue
        result.append(text[i])
        i += 1
    text = "".join(result)

    text = _collapse_horizontal_spaces(text)
    text = _collapse_newlines(text, 2)

    text = _format_sources_section(text)

    return text.strip()


def _format_sources_section(text: str) -> str:
    """Format the Sources section with proper markdown links and structure."""
    sources_idx = -1
    lower_text = text.lower()
    pos = 0
    while True:
        pos = lower_text.find("source", pos)
        if pos == -1:
            break
        j = pos + 6
        if j < len(text) and lower_text[j] == 's':
            j += 1
        k = j
        while k < len(text) and text[k].isspace():
            k += 1
        if k < len(text) and text[k] == ':':
            sources_idx = pos
            break
        elif k == len(text) or text[k] == '\n':
            sources_idx = pos
            break
        pos += 1

    if sources_idx == -1:
        return text

    before_sources = text[:sources_idx]
    sources_section = text[sources_idx:]

    temp = before_sources.rstrip(' \t')
    if temp.endswith('\n**'):
        before_sources = temp[:-2]
    before_sources = before_sources.rstrip() + '\n'

    sources_section = _clean_link_artifacts(sources_section)

    result = []
    i = 0
    n = len(sources_section)
    while i < n:
        j = i
        while j < n and sources_section[j].isspace():
            j += 1
        if j < n and sources_section[j] == '.':
            k = j + 1
            while k < n and sources_section[k].isspace():
                k += 1
            is_domain = False
            matched_dom = ""
            for dom in ("com", "org", "net", "edu", "gov", "io", "co"):
                if sources_section[k:].lower().startswith(dom):
                    dom_end = k + len(dom)
                    if dom_end == n or not sources_section[dom_end].isalnum():
                        is_domain = True
                        matched_dom = dom
                        break
            if is_domain:
                result.append('.')
                result.append(matched_dom)
                i = k + len(matched_dom)
                continue
        result.append(sources_section[i])
        i += 1
    sources_section = "".join(result)

    result = []
    i = 0
    n = len(sources_section)
    while i < n:
        if sources_section[i].isspace() and i + 1 < n and sources_section[i + 1] == '[':
            j = i
            while j < n and sources_section[j].isspace():
                j += 1
            if j < n and sources_section[j] == '[':
                k = j + 1
                while k < n and sources_section[k].isdigit():
                    k += 1
                if k < n and sources_section[k] == ']':
                    result.append('\n')
                    result.append(sources_section[j:k+1])
                    i = k + 1
                    continue
        result.append(sources_section[i])
        i += 1
    sources_section = "".join(result)

    if sources_section.lower().startswith("sources:"):
        sources_section = "Sources:\n" + sources_section[8:].lstrip()
    elif sources_section.lower().startswith("source:"):
        sources_section = "Source:\n" + sources_section[7:].lstrip()

    lines = sources_section.split('\n')
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        stripped_line = line.lower()
        normalized_line = "".join(c for c in stripped_line if c not in ('*', ':')).strip()
        if normalized_line in ("sources", "source"):
            formatted_lines.append('\n**Sources:**\n')
            continue

        if line == '**':
            continue

        if line.startswith('['):
            j = 1
            while j < len(line) and line[j].isdigit():
                j += 1
            if j < len(line) and line[j] == ']':
                formatted_lines.append(_format_single_source(line))
                continue
        
        formatted_lines.append(line)

    return before_sources + '\n'.join(formatted_lines)


def _format_single_source(source_line: str) -> str:
    """Format a single source citation line."""
    if not source_line.startswith('['):
        return source_line
    j = 1
    while j < len(source_line) and source_line[j].isdigit():
        j += 1
    if j >= len(source_line) or source_line[j] != ']':
        return source_line
    num = source_line[1:j]
    rest = source_line[j+1:].strip()

    rest = _clean_link_artifacts(rest)
    rest = _fix_newlines_between_letters(rest)

    has_proper_link = False
    link_idx = 0
    while True:
        link_idx = rest.find("[Link](", link_idx)
        if link_idx == -1:
            break
        after_link = rest[link_idx + 7:]
        if after_link.startswith(("http://", "https://")):
            close_idx = after_link.find(")")
            if close_idx != -1:
                has_proper_link = True
                break
        link_idx += 7

    if has_proper_link:
        return f"\n[{num}] {rest}\n"

    url_start_idx = -1
    for scheme in ("https://", "http://"):
        idx = rest.find(scheme)
        if idx != -1:
            if url_start_idx == -1 or idx < url_start_idx:
                url_start_idx = idx

    if url_start_idx != -1:
        url_end_idx = url_start_idx
        while url_end_idx < len(rest) and rest[url_end_idx] not in ' \t\n\r<>"[]':
            url_end_idx += 1
        url = rest[url_start_idx:url_end_idx]
        url = url.rstrip('.,;:!?)')
        url = "".join(url.split())

        title_part = rest[:url_start_idx].strip()
        title_part = title_part.rstrip('—–- ').strip()
        if title_part.endswith(')'):
            open_p = title_part.rfind('(')
            if open_p != -1:
                title_part = title_part[:open_p].strip()

        domain = ""
        if url.startswith("https://"):
            domain = url[8:]
        elif url.startswith("http://"):
            domain = url[7:]
        if domain.startswith("www."):
            domain = domain[4:]
        slash_idx = domain.find('/')
        if slash_idx != -1:
            domain = domain[:slash_idx]

        after_url = rest[url_end_idx:].strip()
        after_url = after_url.lstrip('—–- ').strip()

        if title_part:
            if after_url:
                return f"\n[{num}] **{title_part}** ({domain}) - [Link]({url})\n    *{after_url}*\n"
            return f"\n[{num}] **{title_part}** ({domain}) - [Link]({url})\n"
        else:
            return f"\n[{num}] [{domain}]({url})\n"

    tlds = (".com", ".org", ".net", ".edu", ".gov", ".io", ".co", ".in", ".uk", ".de", ".fr", ".jp", ".au", ".ca", ".info", ".biz")
    domain = ""
    domain_pos = -1
    idx = 0
    while True:
        open_p = rest.find('(', idx)
        if open_p == -1:
            break
        close_p = rest.find(')', open_p)
        if close_p == -1:
            break
        content = rest[open_p + 1:close_p].strip()
        if content and content[0].isalnum():
            has_tld = False
            for tld in tlds:
                if content.lower().endswith(tld):
                    if all(c.isalnum() or c in '-.' for c in content):
                        has_tld = True
                        break
            if has_tld:
                domain = content
                domain_pos = open_p
                break
        idx = open_p + 1

    if domain:
        title_part = rest[:domain_pos].strip()
        
        clean_title_parts = []
        tp_idx = 0
        while tp_idx < len(title_part):
            op = title_part.find('(', tp_idx)
            if op == -1:
                clean_title_parts.append(title_part[tp_idx:])
                break
            cl = title_part.find(')', op)
            if cl == -1:
                clean_title_parts.append(title_part[tp_idx:])
                break
            clean_title_parts.append(title_part[tp_idx:op])
            content = title_part[op+1:cl]
            is_abbrev = len(content) >= 3 and all(c.isupper() or c == '.' for c in content)
            if not is_abbrev:
                clean_title_parts.append(title_part[op:cl+1])
            tp_idx = cl + 1
        title_part = "".join(clean_title_parts).strip()
        title_part = title_part.rstrip('—–- ').strip()
        
        if title_part.startswith('**') and title_part.endswith('**'):
            title_part = title_part[2:-2]

        after_domain = rest[domain_pos + len(f'({domain})'):].strip()
        after_domain = after_domain.lstrip('—–- ').strip()
        if after_domain.lower().startswith("link"):
            after_domain = after_domain[4:].strip()
        after_domain = after_domain.lstrip('([](').rstrip(')](').strip()

        if after_domain.startswith('('):
            cl = after_domain.find(')')
            if cl != -1:
                rest_after = after_domain[cl+1:].strip()
                rest_after = rest_after.lstrip('—–- ').strip()
                if rest_after.lower().startswith("link"):
                    after_domain = rest_after[4:].strip()
                    after_domain = after_domain.lstrip('([](').rstrip(')](').strip()

        if title_part:
            if after_domain:
                return f"\n[{num}] **{title_part}** ({domain}) - Link\n    *{after_domain}*\n"
            return f"\n[{num}] **{title_part}** ({domain})\n"
        else:
            return f"\n[{num}] {domain}\n"

    return f"\n[{num}] {rest}\n"
