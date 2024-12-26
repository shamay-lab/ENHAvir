import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()

        hidden_dim = 1024
        model_name = "microsoft/deberta-v3-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, inp):
        batch = self.tokenizer(
            inp, max_length=256, padding=True, truncation=True, return_tensors="pt"
        )
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        hidden = self.model(**batch)["last_hidden_state"][:, 0, :]
        preds = self.linear(hidden)
        return preds
        