from pathlib import Path

from common.discovery import discover_chapters

REPO_CHAPTERS = Path(__file__).resolve().parents[1] / "chapters"


def _write(dirpath, name, body):
    (dirpath / name).write_text(body)


GOOD = 'TITLE="T"\nDESCRIPTION="D"\ndef render():\n    pass\n'


def test_real_chapters_dir_lists_chapters_in_order_and_skips_template():
    found = discover_chapters(REPO_CHAPTERS)
    assert [c.name for c in found] == ["beam_shaft", "pump_pipe", "tolerance_stack", "basic_tools"]
    assert [c.title[0] for c in found] == ["1", "2", "3", "4"]                      # the basic tools chapter comes after chapter 3
    assert "_template" not in [c.name for c in found] and all(c.ok for c in found)


def test_order_attribute_places_a_chapter_last_and_bad_values_are_ignored(tmp_path):
    ch = tmp_path / "chapters"; ch.mkdir()
    _write(ch, "a_late.py", GOOD + "ORDER = 90\n")
    _write(ch, "b_default.py", GOOD)
    _write(ch, "c_early.py", GOOD + "ORDER = 10\n")
    _write(ch, "d_bad.py", GOOD + "ORDER = 'last please'\n")
    found = discover_chapters(ch)
    assert [c.name for c in found] == ["c_early", "b_default", "d_bad", "a_late"]         # bad value falls back to 50; ties by filename
    assert [c.order for c in found] == [10.0, 50.0, 50.0, 90.0]


def test_dropping_in_a_file_adds_a_chapter(tmp_path):
    ch = tmp_path / "chapters"; ch.mkdir()
    _write(ch, "a_one.py", GOOD)
    assert [c.name for c in discover_chapters(ch)] == ["a_one"]
    _write(ch, "b_two.py", GOOD.replace('"T"', '"Second"'))
    found = discover_chapters(ch)
    assert [c.name for c in found] == ["a_one", "b_two"]
    assert found[1].title == "Second"


def test_underscore_and_non_py_files_ignored(tmp_path):
    ch = tmp_path / "chapters"; ch.mkdir()
    _write(ch, "_hidden.py", GOOD); _write(ch, "notes.txt", "x")
    assert discover_chapters(ch) == []


def test_broken_chapters_reported_not_fatal(tmp_path):
    ch = tmp_path / "chapters"; ch.mkdir()
    _write(ch, "good.py", GOOD)
    _write(ch, "syntax.py", "def (:\n")
    _write(ch, "missing.py", 'TITLE="x"\n')
    _write(ch, "notcallable.py", 'TITLE="x"\nDESCRIPTION="y"\nrender=5\n')
    by = {c.name: c for c in discover_chapters(ch)}
    assert by["good"].ok
    assert not by["syntax"].ok and "syntax.py" in by["syntax"].error
    assert not by["missing"].ok and "DESCRIPTION" in by["missing"].error
    assert not by["notcallable"].ok


def test_chapters_do_not_import_each_other():
    import re
    for p in REPO_CHAPTERS.glob("*.py"):
        text = p.read_text()
        assert not re.search(r"^\s*(from|import)\s+chapters", text, re.M), p.name
