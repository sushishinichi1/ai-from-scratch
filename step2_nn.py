import torch
import torch.nn as nn

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# 学習データ
# 本当の規則は y = 2x + 1
x = torch.linspace(-10, 10, 1000).reshape(-1, 1).to(device)
y = 2 * x + 1

# とても小さいニューラルネット
model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
).to(device)

# 「予測と正解がどれくらい違うか」
loss_fn = nn.MSELoss()

# 重みを修正する方法
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print()
print("Training start")

for epoch in range(1001):

    # 1. 予測する
    prediction = model(x)

    # 2. 間違いの大きさを計算
    loss = loss_fn(prediction, y)

    # 3. 前回の勾配を消す
    optimizer.zero_grad()

    # 4. どの重みをどう直すべきか計算
    loss.backward()

    # 5. 重みを修正
    optimizer.step()

    if epoch % 200 == 0:
        print(
            f"Epoch {epoch:4d} | Loss: {loss.item():.6f}"
        )

print()
print("Training finished")
print()

# 学習していない値でテスト
test_x = torch.tensor(
    [[3.0], [5.0], [-2.0]],
    device=device
)

with torch.no_grad():
    result = model(test_x)

for input_value, output_value in zip(test_x, result):
    expected = 2 * input_value.item() + 1

    print(
        f"x = {input_value.item():5.1f} "
        f"| prediction = {output_value.item():8.4f} "
        f"| expected = {expected:8.4f}"
    )
