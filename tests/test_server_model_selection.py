import os

from server import choose_model_file, list_model_files, select_model_path


def make_model(path, modified_time):
    path.write_bytes(b"checkpoint")
    os.utime(path, (modified_time, modified_time))
    return path


def test_model_files_are_sorted_newest_first(tmp_path):
    old = make_model(tmp_path / "old.pt", 100)
    newest = make_model(tmp_path / "newest.pt", 300)
    middle = make_model(tmp_path / "middle.pt", 200)
    (tmp_path / "ignore.txt").write_text("not a model", encoding="ascii")

    assert list_model_files(tmp_path) == [newest, middle, old]


def test_model_menu_defaults_to_newest_and_accepts_number(tmp_path):
    old = make_model(tmp_path / "old.pt", 100)
    newest = make_model(tmp_path / "newest.pt", 200)

    assert choose_model_file(tmp_path, input_func=lambda _prompt: "") == newest
    assert choose_model_file(tmp_path, input_func=lambda _prompt: "2") == old


def test_explicit_model_skips_menu(tmp_path):
    model = make_model(tmp_path / "chosen.pt", 100)
    assert select_model_path(model, input_func=lambda _prompt: "invalid") == model
