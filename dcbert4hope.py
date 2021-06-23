import torch
import torch.nn as nn
from transformers import AutoModel


class DcBert4hope(nn.Module):
    def __init__(self, model1, model2, n_classes):
        super(DcBert4hope, self).__init__()
        self.model1 = AutoModel.from_pretrained(model1, return_dict=False)
        self.model2 = AutoModel.from_pretrained(model2, return_dict=False)
        self.fc1 = nn.Linear(self.model1.config.hidden_size, 128)
        self.fc2 = nn.Linear(self.model2.config.hidden_size, 128)
        self.fc3 = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        _, model1_output = self.model1(
            input_ids=input_ids1,
            attention_mask=attention_mask1
        )
        _, model2_output = self.model2(
            input_ids=input_ids2,
            attention_mask=attention_mask2
        )

        model1_out = self.fc1(model1_output)
        model2_out = self.fc2(model2_output)

        merge = torch.cat((model1_out, model2_out), 1)

        act = self.relu(merge)
        out = self.drop(act)
        return self.fc3(out)

