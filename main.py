import torch
import timeit

from torch import nn
from torch.utils.data import DataLoader
from utils.dataLoader import LoadingData
from NN_LSTM.LSTM import LSTM
from NN_LSTM.LSTM import LSTMTrainer, LSTMPredictor
from config import *


# Load datasets
Dataset = LoadingData()
train_dataloader, valid_dataloader, test_dataloader = Dataset.get_train_valid_test_dataloader(time_windows_length=TIME_WINDOWS_LENGTH,
                                                                                              overlap=OVERLAP,
                                                                                              prediction_length=PREDICTION_LENGTH,
                                                                                              train_valid_test_split=TRAIN_VALID_TEST_SPLIT,
                                                                                              num_workers=0,
                                                                                              batch_size=BATCH_SIZE,
                                                                                              shuffle=True)

input, output = next(iter(train_dataloader))


device = torch.device('cpu')  # torch.device('cuda')
model = LSTM(input_size=input.shape[2], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=output.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

trainer = LSTMTrainer(model=model,
                      train_loader=train_dataloader,
                      val_loader=valid_dataloader,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device)



start_time = timeit.default_timer()

trainer.train(num_epochs=EPOCHS)

end_time = timeit.default_timer()
print(f"Training took {end_time - start_time:.2f} seconds.")

# Load the saved model parameters

model.load_state_dict(torch.load('results/models/best_model.pt'))

# Make a prediction with the loaded model
predictor = LSTMPredictor(model=model, device=device)

input, output = next(iter(test_dataloader))
prediction = predictor.predict(input)

print(prediction)
### Code to add if you want to plot the prediction

