import torch
from torch.utils.data import DataLoader
from preprocessing import load_data, preprocess_data, split_data
from build_model import CustomModel

def test_model(test_data, model):
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            outputs = model(input_ids, attention_mask)
            predictions.extend(outputs)

    return predictions

if __name__ == "__main__":
    data = load_data('path/to/data.csv')
    processed_data = preprocess_data(data)
    _, test_data = split_data(processed_data)
    model = CustomModel()
    model.load_state_dict(torch.load('path/to/model.pth'))
    predictions = test_model(test_data, model)
