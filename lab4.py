import torch
import torch.nn as nn
import pandas as pd

class BuyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 5), # 2 входа 5 нейронов
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

class IncomeRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

#n = 1
n = 2 # вариант
df = pd.read_csv('dataset_simple.csv')
if n % 2 == 1:
    # классификация
    X = torch.Tensor(df[['age', 'income']].values)
    y = torch.Tensor(df['will_buy'].values).view(-1, 1)

    model = BuyClassifier()
    loss_fn = nn.BCELoss()
    task_type = 'classification'

else:
    # регрессия
    X = torch.Tensor(df[['age']].values)
    y = torch.Tensor(df['income'].values).view(-1, 1)

    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    model = IncomeRegressor()
    loss_fn = nn.MSELoss()
    task_type = 'regression'

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 200

for epoch in range(0, epochs + 10):
    pred = model(X)
    loss = loss_fn(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print('Ошибка на ' + str(epoch) + ' итерации: ', loss.item())

if task_type == 'classification':
    with torch.no_grad():
        predictions = (model(X) > 0.5).float()
    precision = (predictions == y).float().mean()
    print(f'\nТочность классификации: {precision.item()*100: }%')

else:
    with torch.no_grad():
        predictions = model(X)
    mae = torch.mean(torch.abs(predictions - y)).item()
    print(f'\nСредняя абсолютная ошибка (MAE): {mae:}')