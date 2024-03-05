from common.config import Config
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch

class Reranker:

    def __init__(self, config: Config, device: str):
        self._config = config
        
        rerank_model_name = self._config.rerank_model_name
        self._rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
        self._rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
        self._rerank_model.to(device)
        self._rerank_model.eval()

        self._rerank_top_k = self._config.rerank_top_k
        
    """
    获取关联信息的相关度，用于评判召回质量
    https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker
    """
    def get_reranker_score(self, query: str, reference_results: List[str]):
        """Get reranker score."""
        pairs = []
        for reference in reference_results:
            pairs.append([query, reference])

        with torch.no_grad():
            inputs = self._rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs_on_device = {k: v.to(self._rerank_model.device) for k, v in inputs.items()}
            scores = self._rerank_model(**inputs_on_device, return_dict=True).logits.view(-1, ).float()

        return scores
    
    def top_k(self):
        return self._rerank_top_k