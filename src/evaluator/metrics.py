import re
from typing import List, Dict
from sqlalchemy import create_engine, text
import pandas as pd
import logging
from src.utils import check_sql_syntax, clean_sql_markdown, check_execution_match
import json
import re

logger = logging.getLogger(__name__)

def normalize_sql(sql: str) -> str:
    """
    A simple normalization function for SQL queries.
    """
    
    sql = clean_sql_markdown(sql)

    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    return sql

def check_exact_match(generated_sql: str, ground_truth_sql: str) -> bool:
    """
    Performs a normalized Exact Match (EM) comparison.
    """
    return normalize_sql(generated_sql) == normalize_sql(ground_truth_sql)

class ExperimentEvaluator:
    """
    Handles the evaluation and logging of experiment results.
    """
    def __init__(self, log_file: str, metrics_file: str):
        self.log_file = log_file
        self.metrics_file = metrics_file
        self.results_log = []
        logger.info(f"Evaluator initialized. Logging to {log_file} and {metrics_file}")

    def log_result(self,
                   k_value: int,
                   retriever_name: str,
                   eval_item: dict,
                   generated_sql: str,
                   latency: float,
                   input_tokens: int,
                   output_tokens: int):
        """
        Logs a single prediction and its evaluation.
        """
        ground_truth_sql = eval_item.get('query', '')

        # Đánh giá (giờ sẽ gọi các hàm đã sửa)
        em = check_exact_match(generated_sql, ground_truth_sql)
        syntax_valid = check_sql_syntax(generated_sql)

        # --- 1. CẤU HÌNH KẾT NỐI ---
        # Timeout 5s để tránh treo máy
        db_connection_str = 'postgresql+psycopg2://postgres:password@localhost:5432/cordis_temporary?options=-c search_path=unics_cordis,public -c statement_timeout=5000'
        engine = create_engine(db_connection_str)

        print("🚀 Đang chạy đánh giá...")

        with engine.connect() as conn:
            conn.execute(text("SET search_path TO unics_cordis, public;"))
            status, is_match = check_execution_match(generated_sql, ground_truth_sql, conn)

        log_entry = {
            'k': k_value,
            'retriever': retriever_name,
            'db_id': eval_item.get('db_id', ''),
            'question': eval_item.get('question', ''),
            'ground_truth_sql': ground_truth_sql,
            'generated_sql': generated_sql, 
            'exact_match': em,            
            'syntax_valid': syntax_valid,   
            'execution_status': status,
            'execution_match': is_match,
            'latency_sec': latency,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        }
        self.results_log.append(log_entry)

    def save_results(self):
        """
        Saves the detailed log to CSV and aggregated metrics to JSON.
        """
        if not self.results_log:
            logger.warning("No results to save.")
            return

        # 1. Save detailed log
        log_df = pd.DataFrame(self.results_log)
        try:
            log_df.to_csv(self.log_file, index=False)
            logger.info(f"Detailed logs saved to {self.log_file}")
        except IOError as e:
            logger.error(f"Failed to save logs to {self.log_file}: {e}")

        # 2. Calculate and save aggregated metrics
        agg_metrics = log_df.groupby(['k', 'retriever']).agg(
            total_samples=('question', 'size'),
            exact_match_rate=('exact_match', 'mean'),
            syntax_valid_rate=('syntax_valid', 'mean'),
            execution_match_rate=('execution_match', 'mean'),
            avg_latency_sec=('latency_sec', 'mean'),
            avg_total_tokens=('total_tokens', 'mean')
        ).reset_index()

        agg_metrics['exact_match_rate'] = (agg_metrics['exact_match_rate'] * 100).round(2)
        agg_metrics['syntax_valid_rate'] = (agg_metrics['syntax_valid_rate'] * 100).round(2)
        agg_metrics['execution_match_rate'] = (agg_metrics['execution_match_rate'] * 100).round(2)

        metrics_dict = agg_metrics.to_dict('records')

        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, indent=4)
            logger.info(f"Aggregated metrics saved to {self.metrics_file}")
        except IOError as e:
            logger.error(f"Failed to save metrics to {self.metrics_file}: {e}")

        return metrics_dict

print("File src/evaluator/metrics.py ĐÃ ĐƯỢC SỬA (FIX 21).")
