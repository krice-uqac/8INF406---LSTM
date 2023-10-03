import torch
import torch.nn as nn
import os

from tqdm import tqdm

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

class LSTMTrainer:
        def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
            """
            Initializes the LSTMTrainer class.

            Args:
            - model: the LSTM model to be trained
            - train_loader: the data loader for the training set
            - val_loader: the data loader for the validation set
            - criterion: the loss function to be used
            - optimizer: the optimizer to be used for training
            - device: the device (CPU or GPU) to be used for training
            """
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.criterion = criterion
            self.optimizer = optimizer
            self.device = device
            self.best_val_loss = float('inf')
            self.results_dir = './results/models/'
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)

        def train(self, num_epochs):
            """
            Trains the LSTM model for a specified number of epochs.

            Args:
            - num_epochs: the number of epochs to train the model for
            """
            self.model.to(self.device)
            for epoch in range(num_epochs):
                train_loss = 0.0
                val_loss = 0.0
                self.model.train()
                with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(self.train_loader):.4f}, Val Loss: {val_loss/len(self.val_loader):.4f}") as pbar:
                    for i, (inputs, targets) in enumerate(self.train_loader):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        loss.backward()
                        self.optimizer.step()
                        train_loss += loss.item()
                        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(self.train_loader):.4f}, Val Loss: {val_loss/len(self.val_loader):.4f}")
                        pbar.update(1)

                self.model.eval()
                with torch.no_grad():
                    for i, (inputs, targets) in enumerate(self.val_loader):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        val_loss += loss.item()

                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(self.train_loader):.4f}, Val Loss: {val_loss/len(self.val_loader):.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    checkpoint_path = os.path.join(self.results_dir, f"best_model_epoch_{epoch+1}.pt")
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint at {checkpoint_path}")
            
class LSTMPredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
        return output.cpu().numpy()

