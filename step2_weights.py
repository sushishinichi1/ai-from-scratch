import torch
import torch.nn as nn

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.linspace(-10, 10, 1000).reshape(-1, 1).to(device)
y = 2 * x + 1

model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("=== 学習前 ===")
print("最初の層の重み:")
print(model[0].weight.detach().cpu())

print()
print("最後の層の重み:")
print(model[2].weight.detach().cpu())

for epoch in range(1001):
    prediction = model(x)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print()
print("=== 学習後 ===")
print("最初の層の重み:")
print(model[0].weight.detach().cpu())

print()
print("最後の層の重み:")
print(model[2].weight.detach().cpu())

print()
print("Final loss:", loss.item())
