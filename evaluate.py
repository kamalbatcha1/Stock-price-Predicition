import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def evaluate(model, loader, device):

    model.eval()

    preds, actuals = [], []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            out = model(X)
            prob = torch.sigmoid(out)
            preds.extend(prob.cpu().numpy())
            actuals.extend(y.numpy())

    preds = np.array(preds).flatten()
    print("\nProbability Statistics")
    print("Min:", preds.min())
    print("Max:", preds.max())
    print("Mean:", preds.mean())
    preds = (preds > 0.45).astype(int)
    print("\nPrediction Distribution")
    print(np.unique(preds, return_counts=True))

    print("\nAccuracy:", accuracy_score(actuals, preds))
    print("\nReport:\n", classification_report(actuals, preds))

    return preds