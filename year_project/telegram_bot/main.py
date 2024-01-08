#!/usr/bin/env python
import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

router = Router()

load_dotenv()


@router.message(Command("start"))
async def start(msg: Message) -> None:
    await msg.answer(
        rf"Hi {msg.from_user.full_name}!" +
        "Просто возвращаю твои сообщения обратно")


@router.message()
async def echo(msg: Message) -> None:
    await msg.answer(f"{msg.from_user.full_name} : работаю с твоим сообщением {msg.text}")


async def main() -> None:
    bot = Bot(token=os.environ.get('TOKEN'),
              session=AiohttpSession(proxy='http://proxy.server:3128'))
    if os.environ.get('DISABLE_PROXY'):
        bot = Bot(token=os.environ.get('TOKEN'))

    dp = Dispatcher(bot=bot)
    dp.include_router(router=router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
