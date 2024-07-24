from re import sub
from typing import List, Tuple

from datasets import load_from_disk, load_dataset, Dataset, concatenate_datasets
from faiss import IndexFlatIP
from numpy import array, append
from torch import sum, clamp, no_grad
from tqdm import trange
from transformers import AutoTokenizer, AutoModel


class Finder(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        try:
            self.dataset = load_from_disk('data_embed')
            self.embeddings = array(self.dataset['embeddings'])
        except FileNotFoundError:
            src_ds = load_dataset('IlyaGusev/gazeta', revision="v1.0", trust_remote_code=True, split='train')
            summary = list(map(lambda x: sub(r'«»,:', '', x).lower(), src_ds['summary']))
            embeddings = self.embedding(summary)
            dset_embed = Dataset.from_dict({"embeddings": embeddings})
            dataset = concatenate_datasets([src_ds, dset_embed], axis=1)
            dataset.save_to_disk('data_embed')
        self.index = IndexFlatIP(768)
        self.index.add(self.embeddings)

    def embedding(self, sentences):
        out = array([])
        bs = 500
        for i in trange(0, len(sentences), bs):
            encoded_input = self.tokenizer(sentences[i:i + bs], padding=True, truncation=True, return_tensors='pt').to(
                'cuda:0')
            model = self.model.to('cuda:0')
            with no_grad():
                model_output = model(**encoded_input)
            out = append(out, Finder.mean_pooling(model_output, encoded_input['attention_mask']).cpu())
        return out.reshape((out.size // 768, 768))

    def search(self, sentence: str, k: int = 10) -> List[Tuple[str, str]]:
        embed_sent = self.embedding([self.s_strip(sentence)])
        a = array(embed_sent)
        d, ind = self.index.search(a, k)
        out = []
        for i in ind[0].tolist():
            out.append((self.dataset['summary'][i], self.dataset['url'][i]))
        return out

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return sum(token_embeddings * input_mask_expanded, 1) / clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def s_strip(x: str) -> str:
        return sub(r'«»,:', '', x).lower()
