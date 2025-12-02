import os
import shutil
import subprocess
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_name="cordis_temporary", user="postgres", password="password"):
        self.db_name = db_name
        self.user = user
        self.password = password
        self.db_url = f'postgresql+psycopg2://{user}:{password}@localhost:5432/{db_name}'

    def setup_database(self):
        """Thiết lập user và tạo database."""
        logger.info("Setting up PostgreSQL database...")
        subprocess.run(f"sudo -u postgres psql -c \"ALTER USER postgres PASSWORD '{self.password}';\"", shell=True)
        # Drop nếu tồn tại để reset sạch sẽ
        subprocess.run(f"sudo -u postgres psql -c \"DROP DATABASE IF EXISTS {self.db_name};\"", shell=True)
        subprocess.run(f"sudo -u postgres psql -c \"CREATE DATABASE {self.db_name};\"", shell=True)

    def restore_data(self, source_folder: str, local_dest: str = '/content/cordis_data'):
        """Copy dữ liệu, fix đường dẫn và restore vào DB."""
        logger.info(f"Restoring data from {source_folder}...")

        if os.path.exists(local_dest): shutil.rmtree(local_dest)
        shutil.copytree(source_folder, local_dest)
        os.system(f"chmod -R 777 {local_dest}")

        sql_file = os.path.join(local_dest, "restore.sql")
        fixed_sql = os.path.join(local_dest, "restore_fixed.sql")

        with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Fix path
        content = content.replace("$$PATH$$", local_dest)

        with open(fixed_sql, 'w', encoding='utf-8') as f:
            f.write(content)

        cmd = f"sudo -u postgres psql -d {self.db_name} -f '{fixed_sql}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Restore failed: {result.stderr}")
            raise Exception("Database Restore Failed")
        logger.info("Database restored successfully.")

    def get_engine(self):
        return create_engine(self.db_url + "?options=-c search_path=unics_cordis,public -c statement_timeout=5000")