import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_predictions(model, data_loader):
    model = model.eval()
    sentence = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for data in data_loader:
            texts = data['text']
            input_ids1 = data['input_ids1'].to(device)
            attention_mask1 = data['attention_mask1'].to(device)
            input_ids2 = data['input_ids2'].to(device)
            attention_mask2 = data['attention_mask2'].to(device)
            labels = data["label"].to(device)
            outputs = model(
                input_ids1=input_ids1,
                attention_mask1=attention_mask1,
                input_ids2=input_ids2,
                attention_mask2=attention_mask2
            )

            preds = [0 if x < 0.5 else 1 for x in outputs]
            preds = torch.tensor(preds).to(device)
            sentence.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(labels)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return sentence, predictions, prediction_probs, real_values
