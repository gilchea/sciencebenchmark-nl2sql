import os
import sys
import logging

# Thêm src vào path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các thành phần cần thiết
from src.retriever.semantic import SemanticRetriever
from src.prompt_builder import PromptBuilder
from src.utils import load_json, create_schema_dict

# Tắt bớt log của thư viện
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Cấu hình (Copy từ run_experiment) ---
PROJECT_DIR = "/content/drive/MyDrive/nlp/nl2sql_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data/cordis")
TABLES_FILE = os.path.join(DATA_DIR, "tables.json")
SYNTH_FILE = os.path.join(DATA_DIR, "synth.json") # ICL Pool
DEV_FILE = os.path.join(DATA_DIR, "dev.json")     # Nguồn câu hỏi
RETRIEVER_MODEL = "BAAI/bge-small-en-v1.5"
DEVICE = "cpu" # Dùng CPU cho test nhanh

K_TO_TEST = 3        # Số lượng retrieval
NUM_PROMPTS = 3      # In 3 prompt
# ------------------------------------------

def test_generated_prompts():
    logger.info("--- Bắt đầu Test Prompt Builder ---")

    # 1. Load Data
    tables_data = load_json(TABLES_FILE)
    icl_pool_data = load_json(SYNTH_FILE)
    eval_data = load_json(DEV_FILE)

    # 2. Process Schema
    schema_dict = create_schema_dict(tables_data)

    # 3. Init Retriever
    logger.info(f"Khởi tạo Retriever (model: {RETRIEVER_MODEL})...")
    retriever = SemanticRetriever(model_name=RETRIEVER_MODEL, device=DEVICE)
    valid_icl_pool_data = [ex for ex in icl_pool_data if ex['db_id'] in schema_dict]
    retriever.build_index(valid_icl_pool_data)
    logger.info("Retriever index đã được build.")

    # 4. Init Prompt Builder
    prompt_builder = PromptBuilder()

    # 5. Lấy 3 mẫu đầu tiên từ dev set để test
    test_samples = eval_data[:NUM_PROMPTS]

    for i, item in enumerate(test_samples):
        question = item['question']
        db_id = item['db_id']
        schema_context = schema_dict.get(db_id)

        # a. Retrieve
        logger.info(f"\n--- Đang test prompt #{i+1} (k={K_TO_TEST}) ---")
        logger.info(f"Câu hỏi test: {question}")

        icl_examples = retriever.retrieve(question, K_TO_TEST)
        logger.info(f"Đã retrieve {len(icl_examples)} ví dụ.")

        # b. Build prompt
        prompt = prompt_builder.build(
            schema_context=schema_context,
            question=question,
            icl_examples=icl_examples
        )

        # c. In prompt
        print("\n" + "="*80)
        print(f"PROMPT ĐƯỢC TẠO #{i+1}")
        print("="*80)
        print(prompt) # Đây là kết quả bạn muốn xem
        print("="*80 + "\n")

if __name__ == "__main__":
    test_generated_prompts()
