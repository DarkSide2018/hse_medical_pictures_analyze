import asyncio
import logging
import pickle
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram import F
from tgb_service import (
    create_predictable_dataframe,
    create_predictable_dataframe_cat_boost,
)

from functions import show_models, get_label, get_choice, save_choice

from config_reader import config

bot = Bot(token=config.bot_token.get_secret_value())

logging.basicConfig(level=logging.INFO)

dp = Dispatcher()

models_path = "./models"


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Этот тг бот определяет кожную болезнь по фотографии!\nДля начала выберите модель",
        reply_markup=show_models(models_path),
    )


@dp.message(Command("models"))
async def cmd_models(message: types.Message):
    await message.answer("Выберите модель", reply_markup=show_models(models_path))


@dp.message(F.text.lower() == "xgb")
async def select_xgb(message: types.Message):
    print(f"##### {message.text.lower()}")
    save_choice(message.text.lower())
    await message.reply(
        f'Вы выбрали модель: {message.text.lower()}.\nА теперь загрузите изображение или воспользуйтесь командой "models" для смены модели.',
        reply_markup=types.ReplyKeyboardRemove(),
    )


@dp.message(F.text.lower() == "cat_boost")
async def select_cat_boost(message: types.Message):
    print(f"##### {message.text.lower()}")
    save_choice(message.text.lower())
    await message.reply(
        f'Вы выбрали модель: {message.text.lower()}.\nА теперь загрузите изображение или воспользуйтесь командой "models" для смены модели.',
        reply_markup=types.ReplyKeyboardRemove(),
    )


@dp.message(F.text.lower() == "svm")
async def select_svm(message: types.Message):
    print(f"##### {message.text.lower()}")
    save_choice(message.text.lower())
    await message.reply(
        f'Вы выбрали модель: {message.text.lower()}.\nА теперь загрузите изображение или воспользуйтесь командой "models" для смены модели.',
        reply_markup=types.ReplyKeyboardRemove(),
    )


@dp.message(F.photo)
async def get_prediction(message: types.Message):
    selected_model = get_choice()
    if selected_model == "":
        await message.reply(
            "Выберите для начала модель, а после этого повторно загрузите изображение",
            reply_markup=show_models(),
        )
    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    print(f"### {selected_model}")
    file_path = f"models/{selected_model}.pickle"

    pickled_model = pickle.load(open(file_path, "rb"))

    if selected_model == "cat_boost":
        dataframe = create_predictable_dataframe_cat_boost(downloaded_file)
    else:
        dataframe = create_predictable_dataframe(downloaded_file)

    predicted = pickled_model.predict(dataframe)

    print(f"####### {predicted[0][0]}")

    await message.reply(
        f"{message.from_user.full_name} : предсказанное значение : {get_label(predicted[0][0])}"
    )


@dp.message(Command("help"))
async def show_help(message: types.Message):
    await message.reply(
        "/start для начала\nmodels для выбора модели\nПосле выбора модели загрузите изображение и получите предсказание.\n/info узнать о проекте"
    )


@dp.message(Command("info"))
async def show_info(message: types.Message):
    await message.reply(
        """
    Телеграм бот для проекта - Анализ медицинских изображений\n
    Датасет представляет собой 19500 изображений разбитых на обучающие и тестовые наборы (15500 / 4000).\n
    Данные состоят из изображений 23 типов кожных заболеваний, взятых с http://www.dermnet.com/dermatology-pictures-skin-disease-pictures. \n
    Категории включают прыщи, меланому, экзему, себорейный кератоз, стригущий лишай, буллезную болезнь, ядовитый плющ, псориаз, сосудистые опухоли и т. д.\n
    """
    )


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
