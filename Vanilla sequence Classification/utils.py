from torch.utils.data import DataLoader
import torch
from dataset import KanHope
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_data_loader(df, tokenizer, max_len: object, batch_size):
    ds = KanHope(
        text=df.tweets.to_numpy(),
        label=df.labels.to_numpy(),
        tokenizer=tokenizer ,
    )

    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=2)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_min = int(elapsed_time / 60)
    elapsed_secs: int = int(elapsed_time - (elapsed_min * 60))
    return elapsed_min, elapsed_secs


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['label'].to(device)
        labelsviewed = labels.view(labels.shape[0], 1)

        outputs = model(
            input_ids =input_ids,
            attention_mask=attention_mask,
        )
        _, preds = torch.max(outputs, dim=1)
        preds = [0 if x < 0.5 else 1 for x in outputs]
        preds = torch.tensor(preds).to(device)
        loss = loss_fn(outputs, labelsviewed)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for data in data_loader:
            input_ids  = data['input_ids'].to(device)
            attention_mask  = data['attention_mask'].to(device)
            labels = data['label'].to(device)
            labelsviewed = labels.view(labels.shape[0], 1)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = [0 if x < 0.5 else 1 for x in outputs]
            preds = torch.tensor(preds).to(device)
            loss = loss_fn(outputs, labelsviewed)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            torch.cuda.empty_cache()
    return correct_predictions.double() / n_examples, np.mean(losses)



