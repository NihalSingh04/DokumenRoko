from dotenv import load_dotenv
import os
import requests

load_dotenv()   # .env file load karega

api_key = os.getenv("GROQ_API_KEY")
print("API KEY:", api_key)

url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())