from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util


class CosineSimilarityScorer:
    """
    Uses Sentence-BERT embeddings + cosine similarity to compare
    human vs Gemini annotations (or any pair of text lists).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load a pre-trained sentence embedding model once
        self.model = SentenceTransformer(model_name)

    def score_pairs(
        self,
        human_texts: List[str],
        model_texts: List[str],
    ) -> Tuple[list[float], float, float]:
        """
        human_texts[i] should be the ground-truth annotation
        for the same image as model_texts[i].

        Returns:
            scores: cosine similarity for each pair
            mean_score: average similarity
            std_score: standard deviation
        """
        if len(human_texts) != len(model_texts):
            raise ValueError("human_texts and model_texts must have same length")

        # 1) Encode texts to embeddings
        human_embs = self.model.encode(
            human_texts,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False,
        )
        model_embs = self.model.encode(
            model_texts,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False,
        )

        # 2) Compute cosine similarities
        sim_matrix = util.cos_sim(human_embs, model_embs)

        # 3) Take diagonal (pair i with i)
        scores = sim_matrix.diag().cpu().numpy()

        # 4) Summaries
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        return scores.tolist(), mean_score, std_score
