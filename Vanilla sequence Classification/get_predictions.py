import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_predictions(model, data_loader):
    model = model.eval()
    sentence = []
    predictions = []
    predicted_probs = []
    real_values = []

    with  torch.no_grad():
        for d in data_loader:
            texts = d['text']
            input_ids = d['input_ids'].to(device),
            attention_mask = d['attention_mask'].to(device),
            labels = d['label'].to(device)
            labelsviewed = labels.view(labels.shape[0], 1)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = [0 if x < 0.5 else 1 for x in output]
            preds = torch.tensor(preds).to(device)

            sentence.extend(texts)
            predictions.extend(preds)
            predicted_probs.extend(predicted_probs)
            real_values.extend(real_values)

    predictions = torch.stack(predictions).cpu()
    predicted_probs = torch.stack(predicted_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return sentence, predictions, predicted_probs, real_values
