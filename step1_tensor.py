import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

a = torch.randn(2000, 2000, device=device)
b = torch.randn(2000, 2000, device=device)

torch.cuda.synchronize()

start = time.time()

c = a @ b

torch.cuda.synchronize()

elapsed = time.time() - start

print("Tensor A:", a.shape)
print("Tensor B:", b.shape)
print("Result:", c.shape)
print("Calculation time:", round(elapsed, 4), "seconds")

print(
    "GPU memory used:",
    round(torch.cuda.memory_allocated() / 1024**2, 2),
    "MB"
)
