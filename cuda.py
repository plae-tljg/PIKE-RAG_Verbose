import torch
import gc

# 清理GPU缓存
torch.cuda.empty_cache()

# 强制垃圾回收
gc.collect()

# 再次清理
torch.cuda.empty_cache()

print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")