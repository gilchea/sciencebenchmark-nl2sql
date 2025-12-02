import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils import clean_sql_markdown
import logging

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading model: {self.model_id}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str) -> tuple[str, int, int]:
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
        input_tokens = inputs.input_ids.shape[1]

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id
            )
            output_tokens = outputs[0, input_tokens:].shape[0]
            generated_text = self.tokenizer.decode(outputs[0, input_tokens:], skip_special_tokens=True)

            cleaned_sql = clean_sql_markdown(generated_text)
            if cleaned_sql: cleaned_sql += ';'

            return cleaned_sql, input_tokens, output_tokens
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "-- ERROR", input_tokens, 0