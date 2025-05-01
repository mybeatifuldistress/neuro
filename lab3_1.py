import torch
import random

x = torch.randint(-7, 6, (1, 3))
print(x)

x = x.float()
print(x)

x.requires_grad = True

n = 2
y = pow(x, n)
print(y)

z = y * random.randint(1, 10)
print(z)

m = z.exp()
print(m)

output = m.mean()
output.backward()
print(x.grad)