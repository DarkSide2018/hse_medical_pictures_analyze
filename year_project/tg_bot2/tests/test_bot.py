import sys
import os
from aiogram_tests import MockedBot
from aiogram.filters import Command
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from bot import (
    cmd_start,
    cmd_models,
    select_xgb,
    select_cat_boost,
    select_svm,
)


@pytest.mark.asyncio
async def test_start():
    requester = MockedBot(MessageHandler(cmd_start, Command(commands=["start"])))
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    answer_message = calls.send_message.fetchone().text
    assert (
        answer_message
        == "Привет! Этот тг бот определяет кожную болезнь по фотографии!\nДля начала выберите модель"
    )


@pytest.mark.asyncio
async def test_select_model():
    requester = MockedBot(MessageHandler(cmd_models, Command(commands=["models"])))
    calls = await requester.query(MESSAGE.as_object(text="/models"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Выберите модель"


@pytest.mark.asyncio
async def test_select_xgb():
    requester = MockedBot(MessageHandler(select_xgb, Command(commands=["xgb"])))
    calls = await requester.query(MESSAGE.as_object(text="/xgb"))
    answer_message = calls.send_message.fetchone().text
    assert (
        answer_message
        == 'Вы выбрали модель: /xgb.\nА теперь загрузите изображение или воспользуйтесь командой "models" для смены модели.'
    )


@pytest.mark.asyncio
async def test_select_cat_boost():
    requester = MockedBot(
        MessageHandler(select_cat_boost, Command(commands=["cat_boost"]))
    )
    calls = await requester.query(MESSAGE.as_object(text="/cat_boost"))
    answer_message = calls.send_message.fetchone().text
    assert (
        answer_message
        == 'Вы выбрали модель: /cat_boost.\nА теперь загрузите изображение или воспользуйтесь командой "models" для смены модели.'
    )


@pytest.mark.asyncio
async def test_select_svm():
    requester = MockedBot(MessageHandler(select_svm, Command(commands=["svm"])))
    calls = await requester.query(MESSAGE.as_object(text="/svm"))
    answer_message = calls.send_message.fetchone().text
    assert (
        answer_message
        == 'Вы выбрали модель: /svm.\nА теперь загрузите изображение или воспользуйтесь командой "models" для смены модели.'
    )
