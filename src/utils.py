import json
import random
import numpy as np
import torch
import sqlite3
import logging
import time
import re
from contextlib import contextmanager
import sqlparse # <<< THƯ VIỆN MỚI

import os
import shutil
import subprocess
from sqlalchemy import create_engine, text

import sqlfluff
import logging # Hoặc bất cứ logger nào bạn đang dùng

import pandas as pd
# from sqlalchemy import create_engine, text
import numpy as np

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
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from: {filepath}")
        return []

def save_json(data: dict | list, filepath: str):
    """Saves data to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Data saved to {filepath}")
    except IOError as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")

@contextmanager
def timer(name: str):
    """A simple context manager to measure execution time."""
    start_time = time.time()
    yield
    end_time = time.time()
    logger.info(f"[{name}] executed in {end_time - start_time:.4f} seconds")

def format_schema(db_id: str, tables_data: list) -> str:
    """Formats the schema for a given db_id into a CREATE TABLE string."""
    db_schema = next((db for db in tables_data if db['db_id'] == db_id), None)

    if db_schema is None:
        return f"-- Error: Schema for db_id '{db_id}' not found."

    schema_parts = []
    col_id_to_name = {i: name[1] for i, name in enumerate(db_schema['column_names_original'])}

    for i, table_name in enumerate(db_schema['table_names_original']):
        cols = []
        table_cols = [
            (col_idx, col_name, col_type)
            for col_idx, (col_name_id, col_name) in enumerate(db_schema['column_names_original'])
            if db_schema['column_names'][col_idx][0] == i
            for col_type in [db_schema['column_types'][col_idx]]
        ]

        for col_idx, col_name, col_type in table_cols:
            original_col_name = col_id_to_name[col_idx]
            if original_col_name == '*':
                continue
            cols.append(f"  {original_col_name} {col_type.upper()}")

        pk_cols_indices = [pk_idx for pk_idx in db_schema['primary_keys'] if db_schema['column_names'][pk_idx][0] == i]
        if pk_cols_indices:
            pk_col_names = ", ".join(col_id_to_name[pk_idx] for pk_idx in pk_cols_indices)
            cols.append(f"  PRIMARY KEY ({pk_col_names})")

        table_str = f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n);"
        schema_parts.append(table_str)

    fk_statements = []
    for (from_col_idx, to_col_idx) in db_schema['foreign_keys']:
        from_col_name = col_id_to_name[from_col_idx]
        from_table_idx = db_schema['column_names'][from_col_idx][0]
        from_table_name = db_schema['table_names_original'][from_table_idx]
        to_col_name = col_id_to_name[to_col_idx]
        to_table_idx = db_schema['column_names'][to_col_idx][0]
        to_table_name = db_schema['table_names_original'][to_table_idx]
        fk_statements.append(f"-- {from_table_name}.{from_col_name} TO {to_table_name}.{to_col_name}")

    if fk_statements:
        schema_parts.append("\n-- Foreign Key Relationships:\n" + "\n".join(fk_statements))

    return "\n".join(schema_parts)

def create_schema_dict(tables_data: list) -> dict:
    """Creates a dictionary mapping db_id to its formatted schema string."""
    logger.info("Creating schema string dictionary...")
    schema_dict = {}
    for db in tables_data:
        db_id = db['db_id']
        schema_dict[db_id] = format_schema(db_id, tables_data)
    logger.info(f"Processed {len(schema_dict)} schemas.")
    return schema_dict

def clean_sql_markdown(sql: str) -> str:
    """Dọn dẹp markdown, lỗi LLM, VÀ DẤU CHẤM PHẨY."""
    if not sql:
        return ""

    sql = str(sql).strip()

    # 1. Dọn dẹp Markdown
    if sql.startswith("```sql"):
        sql = sql[6:]
    elif sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]

    sql = sql.strip()

    # 2. Dọn dẹp các lỗi LLM
    if sql.lower().startswith("### solution:"):
        sql = sql[13:]
    if sql.lower().startswith("sql:"):
        sql = sql[4:]

    sql = sql.strip()

    # 3. Cắt bỏ hallucination (lỗi bạn thấy trong log)
    sql_parts = re.split(r'Question:', sql, maxsplit=1, flags=re.IGNORECASE)
    sql = sql_parts[0].strip()

    # 4. Chỉ lấy câu lệnh đầu tiên và BỎ DẤU CHẤM PHẨY (;)
    sql = sql.split(';')[0].strip()

    return sql

# ----- BẮT BUỘC: Import lỗi chính xác -----
# Chúng ta sẽ tìm và import lỗi APIParsingError một cách an toàn
try:
    from sqlfluff.core.errors import APIParsingError
except ImportError:
    try:
        from sqlfluff.api import APIParsingError
    except ImportError:
        # Nếu phiên bản quá cũ/mới, ta bắt lỗi chung nhất
        # Mặc dù điều này hiếm khi xảy ra
        print("Cảnh báo: Không tìm thấy APIParsingError, sẽ bắt Exception chung.")
        APIParsingError = Exception

def check_sql_syntax(sql_query: str) -> bool:
    """
    Kiểm tra xem một câu SQL có cú pháp HỢP LỆ hay không bằng sqlfluff.
    Đây là phương pháp xác thực (validation) đáng tin cậy.
    """
    # 1. Giữ lại logic dọn dẹp của bạn
    sql_query = clean_sql_markdown(sql_query)

    if not sql_query:
        return False

    try:
        # 2. Yêu cầu sqlfluff phân tích câu lệnh
        # 'dialect="ansi"' là chuẩn SQL chung, rất an toàn.
        sqlfluff.parse(sql_query, dialect="ansi")

        # 3. Nếu dòng trên chạy thành công (không văng lỗi)
        #    có nghĩa là cú pháp ĐÚNG.
        return True

    except APIParsingError:
        # 4. Nếu sqlfluff văng lỗi này, có nghĩa là
        #    cú pháp 100% SAI. Đây là điều chúng ta muốn.
        return False

    except Exception as e:
        # 5. Bắt các lỗi không mong muốn khác (lỗi nội bộ, v.v.)
        #    Bạn có thể log lỗi này nếu muốn
        # logger.warning(f"Lỗi sqlfluff không mong muốn: {e} | SQL: {sql_query}")
        return False

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
drive_folder_path = '/content/drive/MyDrive/nlp/nl2sql_project/data/cordis.sql' # Nguồn
local_dest_dir = '/content/cordis_full_data' # Đích (trên Colab)

print("🚀 Bắt đầu quy trình: Copy -> Fix Path -> Restore...")

# --- 2. COPY DỮ LIỆU TỪ DRIVE (Workaround Permission) ---
if os.path.exists(local_dest_dir):
    print(f"🗑️ Xóa thư mục cũ {local_dest_dir} để copy mới...")
    shutil.rmtree(local_dest_dir)

print(f"📂 Đang copy toàn bộ folder từ Drive về {local_dest_dir}...")
try:
    shutil.copytree(drive_folder_path, local_dest_dir)
    print("✅ Copy thành công!")
except Exception as e:
    print(f"❌ Lỗi khi copy: {e}")
    raise e

# Cấp quyền đọc/ghi cho mọi user (để user 'postgres' đọc được)
os.system(f"chmod -R 777 {local_dest_dir}")

# --- 3. [FIX] THAY THẾ $$PATH$$ TRONG FILE SQL ---
sql_file_path = os.path.join(local_dest_dir, "restore.sql")
fixed_sql_path = os.path.join(local_dest_dir, "restore_fixed_new_new.sql")

if os.path.exists(sql_file_path):
    print(f"📝 Tìm thấy file gốc: {sql_file_path}")

    # Đọc nội dung
    with open(sql_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        sql_content = f.read()

    # Thay thế $$PATH$$ bằng đường dẫn thực tế trên Colab
    # Lưu ý: Đôi khi file sql ghi là '$$PATH$$/file.dat', nên ta thay bằng local_dest_dir
    if "$$PATH$$" in sql_content:
        print("🔧 Phát hiện placeholder '$$PATH$$', đang thay thế...")
        fixed_content = sql_content.replace("$$PATH$$", local_dest_dir)
    else:
        print("⚠️ Không thấy '$$PATH$$' trong file, giữ nguyên nội dung nhưng vẫn lưu sang file mới.")
        fixed_content = sql_content

    # Lưu file đã sửa
    with open(fixed_sql_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"✅ Đã tạo file SQL đã sửa lỗi đường dẫn: {fixed_sql_path}")
else:
    print(f"❌ Lỗi nghiêm trọng: Không tìm thấy file {sql_file_path}")
    # Dừng luôn nếu không có file sql
    raise FileNotFoundError("Missing restore.sql")

# --- 4. THỰC THI RESTORE VÀO DATABASE ---
print(f"⏳ Đang nạp dữ liệu từ {fixed_sql_path} vào PostgreSQL...")

# Lưu ý: Chạy lệnh psql
cmd = f"sudo -u postgres psql -d cordis_temporary -f '{fixed_sql_path}'"
restore_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if restore_result.returncode != 0:
    print("❌ Lỗi khi chạy lệnh SQL:")
    print(restore_result.stderr)
else:
    print("✅ Lệnh Restore chạy xong (Kiểm tra dữ liệu bên dưới).")

# # --- 5. KIỂM TRA DỮ LIỆU THỰC TẾ ---
# print("\n🔍 Đang kiểm tra số lượng bản ghi trong Database...")
# # Kết nối DB để check
# db_connection_str = 'postgresql+psycopg2://postgres:password@localhost:5432/cordis_temporary?options=-c search_path=unics_cordis,public'
# engine = create_engine(db_connection_str)

# try:
#     with engine.connect() as conn:
#         # Kiểm tra bảng 'projects' (bảng trung tâm của Cordis)
#         # Nếu bảng này không tồn tại nghĩa là lệnh tạo bảng thất bại
#         result = conn.execute(text("SELECT count(*) FROM projects;"))
#         count = result.scalar()
#         print(f"📊 Số lượng dòng trong bảng 'projects': {count}")

#         if count > 0:
#             print("-" * 50)
#             print("🎉 THÀNH CÔNG RỰC RỠ! DATABASE ĐÃ CÓ DỮ LIỆU.")
#             print("-" * 50)
#         else:
#             print("⚠️ CẢNH BÁO: Bảng có tồn tại nhưng số dòng = 0. Kiểm tra lại nội dung file .dat!")
# except Exception as e:
#     print(f"❌ Lỗi khi query kiểm tra: {e}")
#     print("Gợi ý: Có thể file SQL chưa tạo được bảng. Xem lại log lỗi ở bước 4.")

def check_execution_match(sql_gen: str, sql_truth: str, conn):

    if pd.isna(sql_gen) or str(sql_gen).strip() == "":
        return "Empty Generated SQL", False

    try:
        # Luôn set path để đảm bảo đúng schema
        conn.execute(text("SET search_path TO unics_cordis, public;"))

        # Bọc text() để fix lỗi thư viện
        t_sql_truth = text(sql_truth)
        t_sql_gen = text(sql_gen)

        df_truth = pd.read_sql(t_sql_truth, conn)
        df_gen = pd.read_sql(t_sql_gen, conn)

        if df_truth.shape != df_gen.shape:
            return "Shape Mismatch", False

        if df_truth.empty and df_gen.empty:
            return "Both Empty", True

        vals_truth = df_truth.sort_values(by=df_truth.columns.tolist()).values.tolist()
        vals_gen = df_gen.sort_values(by=df_gen.columns.tolist()).values.tolist()

        if vals_truth == vals_gen:
            return "Match", True
        else:
            return "Value Mismatch", False

    except Exception as e:
        conn.rollback() # Bắt buộc rollback
        err_msg = str(e)
        if "canceling statement due to statement timeout" in err_msg:
            # print(f"⚠️ Dòng {index}: Timeout (>5s)")
            return "Timeout (>5s)", False
        return f"SQL Error: {err_msg.splitlines()[0]}", False
