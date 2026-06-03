import torch
import config


def train(model, loader, criterion, optimizer, device):

    model.train()

    for epoch in range(config.EPOCHS):

        total_loss = 0

        for X, y in loader:

            X, y = X.to(device), y.to(device).unsqueeze(1)

            optimizer.zero_grad()

            preds = model(X)

            loss = criterion(preds, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")