import pandas as pd
from transformers import AutoTokenizer
from utils import create_data_loader
from model import VanillaClassifier
from get_predictions import get_predictions

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test = pd.read_csv('test_hope.csv')

model_name = ['mbert-bert.bin']
model_1 = ['bert-base-uncased',
           # 'distilbert-base-uncased',
           # 'roberta-base',
           # 'xlm-roberta-base',
           # 'bert-large-uncased'
           ]
MAX_LEN = 128
BATCH_SIZE = 32
model = VanillaClassifier(model_1, n_classes=1)
model = model.to(device)
LOAD_MODEL = True
if LOAD_MODEL:
    model.load_state_dict(torch.load(model_name))
tokenizer = AutoTokenizer.from_pretrained(model_1)
test_data_loader = create_data_loader(test, tokenizer, MAX_LEN, BATCH_SIZE)
get_predictions(model, test_data_loader)
