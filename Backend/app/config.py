import os
from dotenv import load_dotenv

load_dotenv('.env.local')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
LOCAL_PRODUCTS_JSON_PATH = "./app/data/products.json"
GITHUB_PRODUCTS_JSON_PATH = "https://raw.githubusercontent.com/ashk12/AI-Phone-Assistant/main/products.json"
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:5173/')