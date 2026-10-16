import torch
import time

SIZE = 3000
REPEATS = 5

print("Matrix size:", SIZE, "x", SIZE)
print()

# -------------------------
# CPU
# -------------------------

a_cpu = torch.randn(SIZE, SIZE)
b_cpu = torch.randn(SIZE, SIZE)

# ウォームアップ
_ = a_cpu @ b_cpu

start = time.time()

for _ in range(REPEATS):
    c_cpu = a_cpu @ b_cpu

cpu_time = (time.time() - start) / REPEATS

print("CPU average:", round(cpu_time, 4), "seconds")


# -------------------------
# GPU
# -------------------------

device = torch.device("cuda")

a_gpu = torch.randn(SIZE, SIZE, device=device)
b_gpu = torch.randn(SIZE, SIZE, device=device)

# GPUの準備運動
_ = a_gpu @ b_gpu
torch.cuda.synchronize()

start = time.time()

for _ in range(REPEATS):
    c_gpu = a_gpu @ b_gpu

torch.cuda.synchronize()

gpu_time = (time.time() - start) / REPEATS

print("GPU average:", round(gpu_time, 4), "seconds")

print()
print("GPU speedup:", round(cpu_time / gpu_time, 2), "x")
print(
    "GPU memory:",
    round(torch.cuda.memory_allocated() / 1024**2, 2),
    "MB"
)
