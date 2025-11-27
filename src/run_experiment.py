import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
import time
import os
import logging

# Import các module tự viết
# Import hàm dọn dẹp
from src.utils import set_seed, load_json, create_schema_dict, timer, clean_sql_markdown
from src.prompt_builder import PromptBuilder
from src.retriever.semantic import SemanticRetriever
from src.evaluator.metrics import ExperimentEvaluator

# Cấu hình logging
logger = logging.getLogger(__name__)

# --- Cấu hình Experiment ---
SEED = 42
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
RETRIEVER_MODEL = "BAAI/bge-small-en-v1.5"
PROJECT_DIR = "/content/drive/MyDrive/nlp/nl2sql_project"

# Cấu hình đường dẫn
DATA_DIR = os.path.join(PROJECT_DIR, "data/cordis")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
TABLES_FILE = os.path.join(DATA_DIR, "tables.json")
SYNTH_FILE = os.path.join(DATA_DIR, "synth.json")
SEED_FILE = os.path.join(DATA_DIR, "seed.json")
DEV_FILE = os.path.join(DATA_DIR, "dev.json")

EVAL_FILE = DEV_FILE
ICL_POOL_FILE = SYNTH_FILE

LOG_FILE = os.path.join(RESULTS_DIR, "logs.csv")
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.json")

# Cấu hình thử nghiệm
K_VALUES = [0, 1, 3, 5]
DEV_SAMPLE_LIMIT = 100 # Giữ 20 mẫu để test
# K_VALUES = [0, 3]
# DEV_SAMPLE_LIMIT = 2 # Giữ 20 mẫu để test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RETRIEVER_NAME = f"semantic_{RETRIEVER_MODEL.split('/')[-1]}"

# ----------------------------

def load_model_and_tokenizer():
    """Loads the Phi-3 model and tokenizer with quantization."""
    logger.info(f"Loading model: {MODEL_ID}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=False
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def generate_sql(model, tokenizer, prompt: str) -> (str, int, int):
    """
    Generates SQL from a given prompt and counts tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(DEVICE)
    input_tokens = inputs.input_ids.shape[1]

    generation_args = {
        "max_new_tokens": 200,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,
        "num_beams": 1,
    }

    try:
        outputs = model.generate(
            **inputs,
            **generation_args
        )

        generated_text = tokenizer.decode(outputs[0, input_tokens:], skip_special_tokens=True)
        output_tokens = outputs[0, input_tokens:].shape[0]

        # === SỬA LỖI (FIX 22): Dọn dẹp ngay tại nguồn ===
        # Dọn dẹp (xóa markdown, ';', và 'Question:...')
        # trước khi trả về
        cleaned_sql = clean_sql_markdown(generated_text)

        # Thêm lại dấu ; vào cuối CÂU LỆNH SẠCH
        if cleaned_sql:
            cleaned_sql += ';'

        return cleaned_sql, input_tokens, output_tokens
        # === KẾT THÚC SỬA LỖI ===

    except Exception as e:
        logger.error(f"Error during model generation: {e}", exc_info=True)
        return f"-- ERROR: {e}", input_tokens, 0


def main():
    """Main function to run the NL-to-SQL experiment."""
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("--- Starting NL-to-SQL Experiment Framework ---")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"K values: {K_VALUES}")

    # 1. Load Model
    with timer("Load Model"):
        model, tokenizer = load_model_and_tokenizer()

    # 2. Load Data
    with timer("Load Data"):
        tables_data = load_json(TABLES_FILE)
        icl_pool_data = load_json(ICL_POOL_FILE) # Dùng synth
        eval_data = load_json(EVAL_FILE)         # Dùng dev

        if not tables_data or not icl_pool_data or not eval_data:
            logger.error("Failed to load one or more data files. Exiting.")
            return

        if DEV_SAMPLE_LIMIT:
            logger.warning(f"Limiting dev set to {DEV_SAMPLE_LIMIT} samples.")
            eval_data = eval_data[:DEV_SAMPLE_LIMIT]

    # 3. Pre-process Schema
    with timer("Pre-process Schema"):
        schema_dict = create_schema_dict(tables_data)

    # 4. Initialize Retriever
    with timer("Initialize Retriever"):
        retriever = SemanticRetriever(model_name=RETRIEVER_MODEL, device=DEVICE)

        if any(k > 0 for k in K_VALUES):
            valid_icl_pool_data = [ex for ex in icl_pool_data if ex['db_id'] in schema_dict]
            logger.info(f"Using {len(valid_icl_pool_data)}/{len(icl_pool_data)} ICL pool examples with valid db_id.")
            retriever.build_index(valid_icl_pool_data)
        else:
            logger.info("Skipping retriever index build (all k=0).")

    # 5. Initialize Prompt Builder & Evaluator
    prompt_builder = PromptBuilder()
    evaluator = ExperimentEvaluator(log_file=LOG_FILE, metrics_file=METRICS_FILE)

    # 6. Run Experiment Loop
    logger.info("--- Starting Experiment Loop ---")

    for k in K_VALUES:
        logger.info(f"Running experiment for k = {k}")

        for eval_item in tqdm(eval_data, desc=f"k={k}"):
            db_id = eval_item['db_id']
            question = eval_item['question']

            schema_context = schema_dict.get(db_id)
            if not schema_context:
                logger.warning(f"No schema found for db_id {db_id}. Skipping item.")
                continue

            # a. Retrieve ICL examples
            icl_examples = []
            if k > 0:
                icl_examples = retriever.retrieve(question, k)

            # b. Build Prompt
            prompt = prompt_builder.build(
                schema_context=schema_context,
                question=question,
                icl_examples=icl_examples
            )

            # c. Generate SQL
            start_time = time.time()
            generated_sql, input_tokens, output_tokens = generate_sql(model, tokenizer, prompt)
            latency = time.time() - start_time

            # d. Log Result
            evaluator.log_result(
                k_value=k,
                retriever_name=RETRIEVER_NAME if k > 0 else "zero-shot",
                eval_item=eval_item,
                generated_sql=generated_sql, # Truyền SQL SẠCH
                latency=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

    # 7. Save Final Results
    logger.info("--- Experiment Finished ---")
    final_metrics = evaluator.save_results()

    logger.info("Aggregated Metrics:")
    print(pd.DataFrame(final_metrics).to_string())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

print("File src/run_experiment.py ĐÃ ĐƯỢC SỬA (FIX 22 - Log Sạch).")
