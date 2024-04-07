import sys
import os
import shutil
from aiogram import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functions import get_list_of_models, show_models, save_choice, get_choice


def file_path():
    return "tests/test_files"


os.makedirs(file_path(), exist_ok=True)
open(f"{file_path()}/model1", "a").close()
open(f"{file_path()}//model2", "a").close()


def test_get_list_of_models_instance():
    assert isinstance(get_list_of_models(file_path()), list)


def test_get_list_of_models_len():
    assert len(get_list_of_models(file_path())) == 2


def test_show_models():
    assert isinstance(show_models(file_path()), types.ReplyKeyboardMarkup)


def test_save_choice():
    model = "model1"
    path = file_path() + "/"
    save_choice(model, path)
    with open(path + "user_choices.txt", "r") as file:
        assert file.read() == model


def test_get_choice():
    model = "model1"

    path = file_path()
    with open(path + "/user_choices.txt", "w") as file:
        file.write(model)
    assert get_choice(path + "/") == model
    shutil.rmtree(path)
