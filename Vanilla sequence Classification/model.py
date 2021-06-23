import torch.nn as nn
from transformers import AutoModel


class VanillaClassifier(nn.Module):
    def __init__(self, model, n_classes):
        super(VanillaClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model, return_dict=False)
        self.fc1 = nn.Linear(self.model.config.hidden_size, 128)
        self.fc2 = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        _, model_output = self.model1(
            input_ids=input_ids1,
            attention_mask=attention_mask1
        )
        model_out = self.fc1(model_output)
        act = self.relu(model_out)
        out = self.drop(act)
        return self.fc3(out)
