import pandas as pd
import sqlfluff
from sqlalchemy import text
from src.utils import clean_sql_markdown
import logging

logger = logging.getLogger(__name__)

# Import lỗi an toàn
try:
    from sqlfluff.core.errors import APIParsingError
except ImportError:
    APIParsingError = Exception

def check_sql_syntax(sql: str) -> bool:
    if not sql: return False
    try:
        sqlfluff.parse(sql, dialect="ansi")
        return True
    except (APIParsingError, Exception):
        return False

def check_execution_match(sql_gen: str, sql_truth: str, conn) -> tuple[str, bool]:
    if not sql_gen or str(sql_gen).strip() == "":
        return "Empty SQL", False

    try:
        conn.execute(text("SET search_path TO unics_cordis, public;"))
        df_truth = pd.read_sql(text(sql_truth), conn)
        df_gen = pd.read_sql(text(sql_gen), conn)

        if df_truth.shape != df_gen.shape:
            return "Shape Mismatch", False

        # So sánh values (đã sort)
        vals_truth = df_truth.sort_values(by=df_truth.columns.tolist()).values.tolist()
        vals_gen = df_gen.sort_values(by=df_gen.columns.tolist()).values.tolist()

        return ("Match", True) if vals_truth == vals_gen else ("Value Mismatch", False)

    except Exception as e:
        conn.rollback()
        err_msg = str(e).splitlines()[0]
        return f"Error: {err_msg}", False

class ExperimentEvaluator:
    def __init__(self, db_engine):
        self.engine = db_engine
        self.logs = []

    def log(self, item, sql_gen, latency, in_tok, out_tok, k, retriever):
        sql_truth = item.get('query', '')
        syntax_ok = check_sql_syntax(sql_gen)

        # Execution Check
        with self.engine.connect() as conn:
            status, is_match = check_execution_match(sql_gen, sql_truth, conn)

        self.logs.append({
            'k': k, 'retriever': retriever,
            'db_id': item['db_id'], 'question': item['question'],
            'generated_sql': sql_gen, 'ground_truth': sql_truth,
            'syntax_valid': syntax_ok, 'exec_status': status,
            'exec_match': is_match, 'latency': latency,
            'tokens': in_tok + out_tok
        })

    def save(self, log_path, metrics_path):
        df = pd.DataFrame(self.logs)
        df.to_csv(log_path, index=False)

        # Aggregate
        metrics = df.groupby(['k', 'retriever']).agg(
            accuracy=('exec_match', 'mean'),
            syntax_valid=('syntax_valid', 'mean'),
            latency=('latency', 'mean')
        ).reset_index()
        metrics.to_json(metrics_path, orient='records', indent=4)
        return metrics