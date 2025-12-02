import faiss
from sentence_transformers import SentenceTransformer
import logging
import torch

logger = logging.getLogger(__name__)

class SemanticRetriever:
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.examples = []
        self.device = device if torch.cuda.is_available() else 'cpu'

    def build_index(self, examples: list):
        self.examples = examples
        if not examples: return

        questions = [ex['question'] for ex in examples]
        embeddings = self.model.encode(questions, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        if self.device == 'cuda':
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except: pass # Fallback CPU

        self.index.add(embeddings)

    def retrieve(self, query: str, k: int) -> list:
        if not self.index or k == 0: return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, I = self.index.search(q_emb, k)
        return [self.examples[i] for i in I[0] if i != -1]