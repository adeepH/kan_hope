import pandas as pd
from transformers import AutoTokenizer
from utils import create_data_loader
from dcbert4hope import DcBert4hope
from get_predictions import get_predictions

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test = pd.read_csv('multichannelhope_test.csv')

model_name = ['mbert-bert.bin']
model1 = ['bert-base-uncased',
          # 'distilbert-base-uncased',
          # 'roberta-base',
          # 'xlm-roberta-base',
          # 'bert-large-uncased'
          ]
model2 = ['bert-base-multilingual-cased',
          # 'distilbert-base-multilingual-cased',
          # 'xlm-roberta-base',
          ]
MAX_LEN = 128
BATCH_SIZE = 32
model = DcBert4hope(model1, model2, n_classes=1)
model = model.to(device)
LOAD_MODEL = True
if LOAD_MODEL:
    model.load_state_dict(torch.load(model_name))
tokenizer1 = AutoTokenizer.from_pretrained(model1)
tokenizer2 = AutoTokenizer.from_pretrained(model2)
test_data_loader = create_data_loader(test, tokenizer1, tokenizer2, MAX_LEN, BATCH_SIZE)
get_predictions(model, test_data_loader)
