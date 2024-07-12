import torch
from torch.utils.data import DataLoader
from transformers import AdamW

from preprocessing import load_data, preprocess_data, split_data
from build_model import CustomModel

def train_model(train_data, model, epochs=3, learning_rate=2e-5):
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    data = load_data('path/to/data.csv')
    processed_data = preprocess_data(data)
    train_data, _ = split_data(processed_data)
    model = CustomModel()
    train_model(train_data, model)
