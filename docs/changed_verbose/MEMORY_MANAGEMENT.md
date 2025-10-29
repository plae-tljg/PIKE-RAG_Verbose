# GPU内存管理模式说明

## 问题背景

在使用IRCoT等需要同时加载LLM模型和embedding模型的任务时，GPU内存可能不够。特别是：
- Qwen 7B模型需要约14GB内存（FP16）
- bge-m3 embedding模型需要约2-3GB内存
- 总计需要约16-17GB，而GPU只有15.47GB

## 解决方案

我们为`QwenClient`添加了两种内存管理模式：

### 1. Persistent模式（持久模式）- 默认

**特点：**
- 模型一直保持在GPU内存中
- **速度快** - 每次调用无需重新加载模型
- **占用更多内存** - 模型常驻GPU

**使用场景：**
- 需要频繁调用模型
- GPU内存充足
- 追求最佳性能

**配置示例：**
```yaml
llm_client:
  module_path: pikerag.llm_client
  class_name: QwenClient
  args:
    memory_mode: "persistent"  # 或者不设置（默认值）
  llm_config:
    model: /path/to/model
```

### 2. Unload After Use模式（用完即卸载）

**特点：**
- 每次使用后自动卸载模型
- **节省内存** - 用完后立即释放GPU内存
- **速度较慢** - 每次使用都需要重新加载模型（需要10-20秒）

**使用场景：**
- GPU内存紧张
- 不频繁调用模型
- 可以接受稍微的延迟

**配置示例：**
```yaml
llm_client:
  module_path: pikerag.llm_client
  class_name: QwenClient
  args:
    memory_mode: "unload_after_use"
  llm_config:
    model: /path/to/model
```

## 如何选择模式

### 推荐使用 Persistent 模式，如果：
- ✅ GPU内存充足（≥16GB）
- ✅ 需要多次调用
- ✅ 追求最佳性能

### 推荐使用 Unload After Use 模式，如果：
- ✅ GPU内存不足（<16GB）
- ✅ 偶尔使用
- ✅ 可以接受加载延迟

## 当前地震任务的配置

在`examples/earthquakes/configs/ircot.yml`中，当前设置为`unload_after_use`模式：

```yaml
llm_client:
  args:
    memory_mode: "unload_after_use"  # 节省内存
```

## Embedding模型

目前embedding模型（bge-m3）还没有类似的卸载功能。如果需要，可以考虑：
1. 使用更小的embedding模型
2. 在CPU上运行embedding模型
3. 或使用CPU embedding和GPU LLM的组合

## 清理GPU内存

如果遇到内存问题，可以运行：

```bash
python cuda.py
```

这会清理PyTorch的CUDA缓存。

## 故障排除

### 问题：CUDA out of memory

**解决方案：**
1. 使用`unload_after_use`模式
2. 减小batch size（如果有）
3. 使用更小的模型
4. 使用更小的embedding模型

### 问题：模型加载太慢

**解决方案：**
1. 使用`persistent`模式（如果内存充足）
2. 检查磁盘速度
3. 将模型文件放在SSD上

## 未来改进

- [ ] 为embedding模型添加类似的内存管理模式
- [ ] 支持partial model unloading（只卸载部分层）
- [ ] 智能内存管理（自动选择最优模式）

