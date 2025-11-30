import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging
import torch

logger = logging.getLogger(__name__)

class SemanticRetriever:
    """
    Retrieves relevant examples based on semantic similarity using SentenceTransformers and FAISS.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cuda'):
        """
        Initializes the retriever and loads the SentenceTransformer model.
        """
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.example_pool = []
        self.device = device if torch.cuda.is_available() else 'cpu'

    def build_index(self, example_pool: List[Dict[str, str]]):
        """
        Builds a FAISS index from a pool of examples.
        """
        self.example_pool = example_pool

        if not example_pool:
            logger.warning("Example pool is empty. Index will not be built.")
            return

        logger.info(f"Building index for {len(example_pool)} examples...")

        questions = [ex['question'] for ex in example_pool]

        embeddings = self.model.encode(questions, convert_to_numpy=True, show_progress_bar=True)

        faiss.normalize_L2(embeddings)

        d = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(d)

        if self.device == 'cuda':
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Successfully moved FAISS index to GPU.")
            except Exception as e:
                logger.warning(f"Failed to move FAISS index to GPU, using CPU. Error: {e}")
                self.device = 'cpu' 

        self.index.add(embeddings)
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def retrieve(self, query_question: str, k: int) -> List[Dict[str, str]]:
        """
        Retrieves the top-k most similar examples for a given query question.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is not built or is empty. Returning no examples.")
            return []

        if k == 0:
            return []

        query_embedding = self.model.encode([query_question], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        D, I = self.index.search(query_embedding, k)

        retrieved_indices = I[0]
        retrieved_examples = [self.example_pool[i] for i in retrieved_indices if i != -1]

        return retrieved_examples
