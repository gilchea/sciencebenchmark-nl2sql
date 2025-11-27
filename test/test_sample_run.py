import torch
import os
import sys
import logging

# Thêm src vào path để import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import set_seed, load_json, create_schema_dict
from src.prompt_builder import PromptBuilder
from src.retriever.semantic import SemanticRetriever
from src.run_experiment import load_model_and_tokenizer, generate_sql, RETRIEVER_MODEL, DEVICE

# Cấu hình logging cơ bản cho test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Cấu hình Test ---
SEED = 42
PROJECT_DIR = "/content/drive/MyDrive/nlp/nl2sql_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data/cordis")
TABLES_FILE = os.path.join(DATA_DIR, "tables.json")
SYNTH_FILE = os.path.join(DATA_DIR, "synth.json") # ICL Pool
DEV_FILE = os.path.join(DATA_DIR, "dev.json")   # Eval Set

# Số lượng mẫu dev để test
TEST_SAMPLE_SIZE = 3
TEST_K = 3 # Chỉ test với k=1

def run_test_pipeline():
    """
    Runs a mini-pipeline on 3 dev samples to ensure everything is connected.
    """
    set_seed(SEED)
    logger.info("--- Starting Test Sample Run ---")

    # 1. Tải 3 mẫu dữ liệu
    try:
        tables_data = load_json(TABLES_FILE)
        synth_data = load_json(SYNTH_FILE)
        dev_data = load_json(DEV_FILE)

        if not tables_data or not synth_data or not dev_data:
            logger.error("Test failed: Could not load data files. Ensure they are in data/")
            return

        test_eval_samples = dev_data[:TEST_SAMPLE_SIZE] # Dùng dev_data
        logger.info(f"Loaded {len(test_eval_samples)} eval samples for testing.")

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # 2. Tải Model (Phần tốn thời gian nhất)
    logger.info("Loading model (this may take a moment)...")
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    # 3. Chuẩn bị tools
    logger.info("Setting up schema, retriever, and builder...")
    schema_dict = create_schema_dict(tables_data)

    retriever = SemanticRetriever(model_name=RETRIEVER_MODEL, device=DEVICE)
    valid_synth_data = [ex for ex in synth_data if ex['db_id'] in schema_dict]
    retriever.build_index(valid_synth_data) # Build index trên synth_data

    prompt_builder = PromptBuilder()

    logger.info("--- Running Test Generations ---")

    for i, item in enumerate(test_eval_samples):
        db_id = item['db_id']
        question = item['question']
        ground_truth = item['query']

        logger.info(f"\n--- Sample {i+1}/{TEST_SAMPLE_SIZE} ---")
        logger.info(f"DB_ID: {db_id}")
        logger.info(f"Question: {question}")
        logger.info(f"Ground Truth: {ground_truth}")

        # 4. Lấy Schema
        schema_context = schema_dict.get(db_id)
        if not schema_context:
            logger.warning("Schema not found, skipping.")
            continue

        # 5. Retrieve
        icl_examples = retriever.retrieve(question, k=TEST_K)
        logger.info(f"Retrieved {len(icl_examples)} ICL example(s).")
        if icl_examples:
             logger.info(f"  > Example 1: {icl_examples[0]['question']}")

        # 6. Build Prompt
        prompt = prompt_builder.build(schema_context, question, icl_examples)

        # 7. Generate
        generated_sql, _, _ = generate_sql(model, tokenizer, prompt)

        print(f"\n[Generated SQL]:")
        print(generated_sql)
        print("-" * 20)

    logger.info("--- Test Sample Run Finished ---")
    logger.info("Nếu bạn thấy 3 câu SQL được sinh ra ở trên, pipeline đã hoạt động!")

if __name__ == "__main__":
    run_test_pipeline()

print("File test/test_sample_run.py đã được ghi.")
