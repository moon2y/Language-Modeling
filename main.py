import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt

from dataset import Shakespeare
from model import CharRNN, CharLSTM
from generate import generate

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        hidden = model.init_hidden(batch_size)
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)
        
        optimizer.zero_grad()
        
        if isinstance(hidden, tuple):
            hidden = tuple(h.data for h in hidden)
        else:
            hidden = hidden.data
        
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    trn_loss = total_loss / len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            hidden = model.init_hidden(batch_size)
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)
            else:
                hidden = hidden.to(device)
            
            if isinstance(hidden, tuple):
                hidden = tuple(h.data for h in hidden)
            else:
                hidden = hidden.data
            
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            
            total_loss += loss.item()
    
    val_loss = total_loss / len(val_loader)
    return val_loss

def main():
    input_file = '/content/drive/MyDrive/인공지능활용 과제/shakespeare_train.txt'
    batch_size = 64
    hidden_size = 256
    n_layers = 2
    learning_rate = 0.001
    n_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Shakespeare(input_file)
    
    # Split the dataset into training and validation sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    trn_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    
    # Train and evaluate CharRNN
    rnn_model = CharRNN(input_size=len(dataset.chars), hidden_size=hidden_size, output_size=len(dataset.chars), n_layers=n_layers)
    rnn_model.to(device)
    rnn_criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    
    rnn_train_losses = []
    rnn_val_losses = []

    for epoch in range(n_epochs):
        trn_loss = train(rnn_model, trn_loader, device, rnn_criterion, rnn_optimizer)
        val_loss = validate(rnn_model, val_loader, device, rnn_criterion)
        
        rnn_train_losses.append(trn_loss)
        rnn_val_losses.append(val_loss)
        
        print(f'RNN Epoch {epoch+1}/{n_epochs}, Training Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    # Train and evaluate CharLSTM
    lstm_model = CharLSTM(input_size=len(dataset.chars), hidden_size=hidden_size, output_size=len(dataset.chars), n_layers=n_layers)
    lstm_model.to(device)
    lstm_criterion = nn.CrossEntropyLoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
    
    lstm_train_losses = []
    lstm_val_losses = []

    for epoch in range(n_epochs):
        trn_loss = train(lstm_model, trn_loader, device, lstm_criterion, lstm_optimizer)
        val_loss = validate(lstm_model, val_loader, device, lstm_criterion)
        
        lstm_train_losses.append(trn_loss)
        lstm_val_losses.append(val_loss)
        
        print(f'LSTM Epoch {epoch+1}/{n_epochs}, Training Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    # Plotting the losses
    plt.figure(figsize=(12, 6))
    plt.plot(rnn_train_losses, label='RNN Training Loss')
    plt.plot(rnn_val_losses, label='RNN Validation Loss')
    plt.plot(lstm_train_losses, label='LSTM Training Loss')
    plt.plot(lstm_val_losses, label='LSTM Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses for RNN and LSTM')
    plt.legend()
    plt.show()

    # Select the model with the best validation performance
    best_model = lstm_model if min(lstm_val_losses) < min(rnn_val_losses) else rnn_model
    best_model_name = "LSTM" if best_model == lstm_model else "RNN"

    # Generate samples using the best model
    seed_characters_list = ["ROMEO:", "JULIET:", "HAMLET:", "OTHELLO:", "MACBETH:"]
    temperatures = [0.5, 0.7, 1.0, 1.2, 1.5]
    gen_length = 100

    for seed_characters in seed_characters_list:
        for temp in temperatures:
            generated_text = generate(best_model, seed_characters, temp, dataset.char_to_idx, dataset.idx_to_char, device, gen_length)
            print(f"\nGenerated text with {best_model_name} (Temperature {temp}):\n{generated_text}\n")

if __name__ == '__main__':
    main()


