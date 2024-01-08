import os
import requests

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get('TOKEN')
chat_id = -4040171880

print(BOT_TOKEN)

url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id=-4040171880'

_ = requests.post(url, json={'text': 'Работаем с телеграм апи'}, timeout=10)