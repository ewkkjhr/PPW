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
    
    # def forward_no_sigmoid(self, x):
    #     x = torch.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    
    def train(self,X,y_ture):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # y = np.copy(y_ture)
        # y[y != 1] = 0
        # 将数据转换为 PyTorch 张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_ture, dtype=torch.float32).view(-1, 1).to(device)
        d = X.shape[1]
        self.fc1 = nn.Linear(d, 8).to(device)  # 输入层到隐藏层
        self.fc2 = nn.Linear(8, 1).to(device)  # 隐藏层到输出层
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.1)
        patience = 5  # 设置提前停止的耐心值
        best_loss = float('inf')  # 初始化最佳损失为无穷大
        counter = 0  # 初始化计数器
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
                # print(f'Early stopping triggered after epoch {epoch + 1}')
                break

        with torch.no_grad():
            for weight in self.parameters():
                # self.theta.append(weight.detach().clone().numpy())
                self.theta.append(weight.detach().cpu().clone().numpy())

    def predict(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将输入数据 x 转换为张量
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction = self.forward(x_tensor)
        return prediction.cpu().numpy()  # For regression, return the continuous output directly
        # # 使用训练好的模型进行预测
        # with torch.no_grad():
        #     prediction = self.forward(x_tensor)
        #     score = self.forward_no_sigmoid(x_tensor)
        # prediction = prediction.numpy()
        # prediction[prediction>0.5] = 1
        # prediction[prediction <= 0.5] = -1
        # prediction = prediction.astype(int)
        # return score.numpy(),prediction