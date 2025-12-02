import os
import logging
import torch
from tqdm import tqdm
import time
import pandas as pd

# Import các module từ src
from src.utils import set_seed, load_json
from src.database.manager import DatabaseManager
from src.datasets.schema import create_schema_dict
from src.models.llm import LLMEngine
from src.retriever.semantic import SemanticRetriever
from src.prompts.builder import PromptBuilder
from src.metrics.evaluator import ExperimentEvaluator


# Cấu hình
CONFIG = {
    "seed": 42,
    "model_id": "microsoft/Phi-3-mini-4k-instruct",
    "retriever_model": "BAAI/bge-small-en-v1.5",
    "data_dir": "/content/drive/MyDrive/nlp/nl2sql_project/data/cordis", # Sửa lại path data của bạn
    "sql_source": "/content/drive/MyDrive/nlp/nl2sql_project/data/cordis.sql",
    # "data_dir": os.path.join(PROJECT_ROOT, "data/cordis"),
    # "sql_source": os.path.join(PROJECT_ROOT, "data/cordis.sql"),

    # "results_dir": os.path.join(PROJECT_ROOT, "results"),
    "k_values": [0, 1, 3],
    "sample_limit": 15, # Test 50 mẫu
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

logger = logging.getLogger(__name__)

def main():
    set_seed(CONFIG["seed"])

    # 1. Setup Database
    db_manager = DatabaseManager()
    db_manager.setup_database()
    try:
        db_manager.restore_data(CONFIG["sql_source"])
    except Exception:
        logger.warning("DB Restore failed or skipped. Execution metrics might fail.")

    # 2. Load Data & Models
    tables = load_json(os.path.join(CONFIG["data_dir"], "tables.json"))
    synth_data = load_json(os.path.join(CONFIG["data_dir"], "synth.json"))
    dev_data = load_json(os.path.join(CONFIG["data_dir"], "dev.json"))[:CONFIG["sample_limit"]]

    schema_map = create_schema_dict(tables)

    llm = LLMEngine(CONFIG["model_id"], CONFIG["device"])
    retriever = SemanticRetriever(CONFIG["retriever_model"], CONFIG["device"])
    retriever.build_index([ex for ex in synth_data if ex['db_id'] in schema_map])

    prompt_builder = PromptBuilder()
    evaluator = ExperimentEvaluator(db_manager.get_engine())

    # 3. Experiment Loop
    logger.info("--- Starting Experiment ---")
    for k in CONFIG["k_values"]:
        logger.info(f"Processing k={k}")
        for item in tqdm(dev_data):
            schema = schema_map.get(item['db_id'])
            if not schema: continue

            # Retrieve & Build Prompt
            icl_ex = retriever.retrieve(item['question'], k) if k > 0 else []
            prompt = prompt_builder.build(schema, item['question'], icl_ex)

            # Generate
            start = time.time()
            sql, in_tok, out_tok = llm.generate(prompt)
            latency = time.time() - start

            # Log
            evaluator.log(item, sql, latency, in_tok, out_tok, k, "semantic")

    # 4. Save Results
    metrics = evaluator.save("results/logs.csv", "results/metrics.json")
    print("\nFinal Metrics:\n", metrics)

if __name__ == "__main__":
    main()