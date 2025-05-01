import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
print(df.head())

y = df.iloc[:,4].values
y = np.where(y=="Iris-setosa",1.,-1.)

y=torch.from_numpy(y)
x = torch.from_numpy(df.iloc[:,[0,2,3]].values)

x = x.to(dtype=torch.float32)
y = y.to(dtype=torch.float32)
print(x)

plt.figure()
plt.scatter(x[y==1., 0], x[y==1., 1], color='red', marker='o')
plt.scatter(x[y==-1., 0], x[y==-1., 1], color='blue', marker='x')
plt.xlabel("X")
plt.ylabel("Y")

# создадим 3 сумматора без функци активации, это называется полносвязный слой (fully connected layer)
# Отсутствие фунций активаци на выходе сумматора эквивалетно наличию  линейной активации
linear = nn.Linear(3, 1)

# при создании веса и смещения инициализируются автоматически
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# выберем вид функции ошибки и оптимизатор
# фунция ошибки показывает как сильно ошибается наш алгоритм в своих прогнозах
lossFn = nn.MSELoss() # MSE - среднеквадратичная ошибка, вычисляется как sqrt(sum(y^2 - yp^2))

# создадим оптимизатор - алгоритм, который корректирует веса наших сумматоров (нейронов)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.003) # lr - скорость обучения

# прямой проход (пресказание) выглядит так:
yp = linear(x)

# имея предсказание можно вычислить ошибку
loss = lossFn(yp, y)
print('Ошибка: ', loss.item())

# и сделать обратный проход, который вычислит градиенты (по ним скорректируем веса)
loss.backward()

# градиенты по параметрам
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# далее можем сделать шаг оптимизации, который изменит веса 
# на сколько изменится каждый вес зависит от градиентов и скорости обучения lr
optimizer.step()

xl=np.linspace(min(x[:,0]), max(x[:,0])) # диапазон координаты x для построения линии
xl=torch.from_numpy(xl)
# построим сначала данные на плоскости
plt.figure
plt.scatter(x[y==1., 0], x[y==1., 1], color='red', marker='o')
plt.scatter(x[y==-1., 0], x[y==-1., 1], color='blue', marker='x') 

y=y.view(100,1)
# итерационно повторяем шаги
# в цикле (фактически это и есть алгоритм обучения):
for i in range(0,100):
    w = linear.weight.detach().numpy().copy()
    yl = -(xl*w[0][0]+xl*w[0][1]+linear.bias.detach().numpy().copy()[0])/w[0][2]
    if((i+1)%10==0):
        plt.plot(xl, yl)
        plt.text(xl[-1], yl[-1], i+1, dict(size=10, color='gray'))
        plt.pause(1)
    pred = linear(x)
    loss = lossFn(pred, y)
    print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())
    loss.backward()
    optimizer.step()
    
pred = torch.Tensor(np.where(pred >=0, 1, -1).reshape(-1,1))
err = sum(abs(y-pred))/2
print('\nОшибка (количество несовпавших ответов): ')
print(err)

plt.text(xl[-1]-0.3, yl[-1], 'END', dict(size=14, color='red'))
plt.show() 