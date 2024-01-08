#!/usr/bin/env python
import asyncio
import logging
import os
import pickle

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

from year_project.telegram_bot.tgb_service import create_predictable_dataframe, get_label
from year_project.telegram_bot.upload_model_to_s3 import model_from_s3

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

router = Router()

load_dotenv()

bot = Bot(token=os.environ.get('TOKEN'))


@router.message(Command("start"))
async def start(msg: Message) -> None:
    await msg.answer(
        rf"Hi {msg.from_user.full_name}!" +
        "Предсказываю кожную болезнь по картинке")


@router.message(Command("help"))
async def echo(msg: Message) -> None:
    await msg.answer(f"{msg.from_user.full_name} : работаю с твоим сообщением {msg.text}")


@router.message(Command("send_photo_svm"))
async def handle_image_svm(message: types.Message):
    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    pickled_model = pickle.load(open('svm_svc_best_estimator.pickle', 'rb'))
    dataframe = create_predictable_dataframe(downloaded_file)
    predicted = pickled_model.predict(dataframe)
    await message.answer(f"{message.from_user.full_name} : предсказанное значение : {get_label(predicted[0])}")


@router.message(Command("send_photo_cat_boost"))
async def handle_image_cat_boost(message: types.Message):
    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    file_path = 'gs_cb_best_estimator.pickle'
    if os.path.exists(file_path):
        print("file exists")
    else:
        print("file does not exist")
        model_from_s3("gs_cb_best_estimator")
    pickled_model = pickle.load(open(file_path, 'rb'))
    dataframe = create_predictable_dataframe(downloaded_file)
    predicted = pickled_model.predict(dataframe)
    await message.answer(f"{message.from_user.full_name} : предсказанное значение : {get_label(predicted[0][0])}")


@router.message(Command("send_photo_xg_boost"))
async def handle_image_xg_boost(message: types.Message):
    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    pickled_model = pickle.load(open('xgb_classifier.pickle', 'rb'))
    dataframe = create_predictable_dataframe(downloaded_file)
    predicted = pickled_model.predict(dataframe)
    await message.answer(f"{message.from_user.full_name} : предсказанное значение : {get_label(predicted[0])}")


async def main() -> None:
    dp = Dispatcher(bot=bot)
    dp.include_router(router=router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
