import os
import requests

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get('TOKEN_NOTE')
chat_id = os.environ.get('chat_id')


print(BOT_TOKEN)

url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={chat_id}'


# _ = requests.post(url, json={'text': 'Работаем с телеграм апи'}, timeout=10)

def notify_bot(message):
    _ = requests.post(url, json={'text': f'{message}'}, timeout=10)
