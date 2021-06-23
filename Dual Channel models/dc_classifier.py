import time
import torch
import pandas as pd
from sklearn.metrics import classification_report
from dcbert4hope import DcBert4hope
from utils import create_data_loader, train_epoch, eval_model, epoch_time
import torch.nn as nn
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup
from collections import defaultdict
from get_predictions import get_predictions

# Set the path of the Kannada Hope speech dataset here
# Reading the previously split train, test, and validation dataframes
train = pd.read_csv('multichannelhope.csv')
val = pd.read_csv('multichannelhope_test.csv')
test = pd.read_csv('multichannelhope_test.csv')
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in pretrained_models]
tokenizer1 = AutoTokenizer.from_pretrained(model1)
tokenizer2 = AutoTokenizer.from_pretrained(model2)
BATCH_SIZE = 32
MAX_LEN = 128
loss_fn = nn.CrossEntropyLoss.to(device)
classification_reports = []
for tokenizer, pretrained_model in zip(model1, model2):
    tokenizer1 = AutoTokenizer.from_pretrained(model1)
    tokenizer2 = AutoTokenizer.from_pretrained(model2)
    train_data_loader = create_data_loader(train, tokenizer1, tokenizer2, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(val, tokenizer1, tokenizer2, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test, tokenizer1, tokenizer2, MAX_LEN, BATCH_SIZE)

    model = DcBert4hope(model1, model2, 1)

    model = model.to(device)

    EPOCHS = 5
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup()
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            scheduler,
            train.shape[0]
        )

        end_time = time.time()
        epoch_min, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_min}m {epoch_secs}s')
        print(f'Train Acc1 {train_acc} Train loss {train_loss}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            val.shape[0]
        )
        print(f'Val Acc1 {val_acc} Val Loss {val_loss}')

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )
    classes = ['Not-hope', 'Hope']
    classification_reports.append(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
