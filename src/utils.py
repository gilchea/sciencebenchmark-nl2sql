import json
import random
import numpy as np
import torch
import logging
import time
import re
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed_value: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logger.info(f"Set seed to {seed_value}")

def load_json(filepath: str) -> list | dict:
    """Loads a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return []

def save_json(data, filepath: str):
    """Saves data to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")

@contextmanager
def timer(name: str):
    start_time = time.time()
    yield
    end_time = time.time()
    logger.info(f"[{name}] executed in {end_time - start_time:.4f} seconds")

def clean_sql_markdown(sql: str) -> str:
    """Dọn dẹp markdown, lỗi LLM, và dấu chấm phẩy."""
    if not sql: return ""
    sql = str(sql).strip()

    # Dọn markdown
    if sql.startswith("```sql"): sql = sql[6:]
    elif sql.startswith("```"): sql = sql[3:]
    if sql.endswith("```"): sql = sql[:-3]

    # Dọn prefix thừa
    if sql.lower().startswith("### solution:"): sql = sql[13:]
    if sql.lower().startswith("sql:"): sql = sql[4:]

    # Cắt hallucination
    sql_parts = re.split(r'Question:', sql, maxsplit=1, flags=re.IGNORECASE)
    sql = sql_parts[0].strip()

    # Bỏ dấu chấm phẩy cuối để xử lý thống nhất sau này
    sql = sql.split(';')[0].strip()
    return sql