from app.core.text_markup import (
    _find_tagged_spans,
    _match_tag,
    extract_tagged_sections,
    remove_tagged_sections,
    replace_fenced_code_blocks,
)


def test_match_tag_basic_open():
    assert _match_tag("<think>", 0, {"think"}) == ("think", False, 7)


def test_match_tag_closing():
    assert _match_tag("</think>", 0, {"think"}) == ("think", True, 8)


def test_match_tag_rejects_unknown():
    assert _match_tag("<b>", 0, {"think"}) is None


def test_match_tag_rejects_plain_text():
    assert _match_tag("hello", 0, {"think"}) is None


def test_extract_tagged_sections_basic():
    text = "pre <reason>because</reason> post"
    extracted, cleaned = extract_tagged_sections(text, ["reason"])
    assert extracted == ["because"]
    assert cleaned == "pre  post"


def test_extract_tagged_sections_nested_keeps_top_level():
    text = "<a>outer <a>inner</a></a>"
    extracted, _ = extract_tagged_sections(text, ["a"])
    assert extracted == ["outer <a>inner</a>"]


def test_extract_tagged_sections_empty_when_none():
    extracted, cleaned = extract_tagged_sections("no tags here", ["reason"])
    assert extracted == []
    assert cleaned == "no tags here"


def test_remove_tagged_sections():
    assert remove_tagged_sections("x <r>secret</r> y", ["r"]) == "x  y"


def test_replace_fenced_code_blocks_no_fences():
    assert replace_fenced_code_blocks("plain text", "[code]") == "plain text"


def test_replace_fenced_code_blocks_single():
    assert replace_fenced_code_blocks("a ```py\nx\n``` b", "[code]") == "a [code] b"


def test_replace_fenced_code_blocks_multiple():
    out = replace_fenced_code_blocks("```a``` mid ```b```", "[c]")
    assert out.count("[c]") == 2
