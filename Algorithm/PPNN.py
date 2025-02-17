import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PerPreNN(nn.Module):
    def __init__(self,max_iter = 10000):
        super(PerPreNN, self).__init__()
        self.max_iter = max_iter
        self.theta = []

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train(self,X,y_ture):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_ture, dtype=torch.float32).view(-1, 1).to(device)
        d = X.shape[1]
        self.fc1 = nn.Linear(d, 8).to(device)  
        self.fc2 = nn.Linear(8, 1).to(device)  
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.1)
        patience = 5 
        best_loss = float('inf') 
        counter = 0  
        self.to(device)

        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            outputs = self.forward(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            val_outputs = self.forward(X_tensor)
            val_loss = criterion(val_outputs, y_tensor)
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0 
            else:
                counter += 1 
            if counter >= patience:
                break

        with torch.no_grad():
            for weight in self.parameters():
                self.theta.append(weight.detach().cpu().clone().numpy())

    def predict(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction = self.forward(x_tensor)
        return prediction.cpu().numpy() 
