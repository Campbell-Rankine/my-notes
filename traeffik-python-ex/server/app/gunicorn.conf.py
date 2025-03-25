import multiprocessing
from dotenv import load_dotenv
import os

load_dotenv("./.env")

bind = os.environ["ADDRESS"]
workers = os.environ["GUNICORN_WORKERS"]
