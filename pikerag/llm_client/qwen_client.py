# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 安全导入 torch_musa，避免在非MUSA系统上出错
try:
    import torch_musa  # noqa: F401
    TORCH_MUSA_AVAILABLE = True
except ImportError:
    TORCH_MUSA_AVAILABLE = False


def get_torch_dtype(type_str: str) -> torch.dtype:
    type_str = type_str.strip().lower()
    if type_str.startswith("torch."):
        type_str = type_str[6:]
    try:
        torch_dtype = getattr(torch, type_str)
        return torch_dtype
    except AttributeError as exc:
        raise ValueError(f"Unrecognized torch.dtype: {type_str}") from exc


class QwenClient(BaseLLMClient):
    NAME = "QwenClient"
    """
    通用的 HuggingFace 本地模型客户端，支持各种开源模型如：
    - DeepSeek-Coder 系列（代码专用）
    - WizardCoder 系列
    - StarCoder 系列
    - Qwen 系列
    - 其他兼容的 transformers 模型
    """

    def __init__(
        self, location: str = None, auto_dump: bool = True, logger: Logger=None, llm_config: dict = None,
        max_attempt: int = 5, exponential_backoff_factor: int = None, unit_wait_time: int = 60, 
        memory_mode: str = "persistent", **kwargs,
    ) -> None:
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)

        assert "model" in llm_config, "`model` should be provided in `llm_config` to initialize `QwenClient`!"
        self._model_id: str = llm_config["model"]
        
        # Memory management mode: "persistent" or "unload_after_use"
        # - persistent: model stays in GPU memory (default, faster but uses more memory)
        # - unload_after_use: model is unloaded after each use (slower but saves memory)
        self._memory_mode = memory_mode
        self._client = None
        self._tokenizer = None
        self._model_kwargs = kwargs
        self._is_musa_gpu = False
        
        # Initialize model immediately for persistent mode
        if self._memory_mode == "persistent":
            self._init_agent(**kwargs)

    def _init_agent(self, **kwargs) -> None:
        """Initialize or reload the model. Safe to call multiple times."""
        if self._client is not None:
            # Model already loaded
            return
            
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_musa_gpu = False
        
        # 检查MUSA GPU可用性
        if TORCH_MUSA_AVAILABLE and hasattr(torch, 'musa') and torch.musa.is_available():
            device = "musa"
            is_musa_gpu = True
            self._is_musa_gpu = True
            
            # 设置环境变量强制使用fp32
            import os
            os.environ["TORCH_MUSA_FORCE_FP32"] = "1"
            os.environ["MUSA_FORCE_FP32"] = "1"
        
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
        
        # 设置模型参数
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        # MUSA GPU特殊配置
        if is_musa_gpu:
            # MUSA GPU特殊处理：先在CPU上加载，然后移动到MUSA GPU
            model_kwargs["device_map"] = None  # 禁用自动设备映射
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["dtype"] = torch.float32
            # 禁用所有可能导致fp16的选项
            model_kwargs["attn_implementation"] = "eager"  # 避免使用flash attention
            model_kwargs["use_cache"] = False  # 禁用缓存
        else:
            model_kwargs["device_map"] = "auto" if device != "cpu" else None
        
        # 处理数据类型
        if "torch_dtype" in kwargs or "dtype" in kwargs:
            # 如果用户指定了数据类型，但使用的是MUSA GPU，强制使用fp32
            if is_musa_gpu:
                model_kwargs["dtype"] = torch.float32
                print("MUSA GPU detected, forcing fp32 precision")
            else:
                # 使用新的dtype参数名
                dtype_key = "dtype" if "dtype" in kwargs else "torch_dtype"
                model_kwargs["dtype"] = get_torch_dtype(kwargs[dtype_key])
        else:
            # 自动选择数据类型
            if is_musa_gpu:
                # MUSA GPU只能使用fp32
                model_kwargs["dtype"] = torch.float32
                print("MUSA GPU detected, using fp32 precision")
            elif device != "cpu":
                # 对于其他GPU，使用 float16 可以节省内存
                model_kwargs["dtype"] = torch.float16
        
        # 合并其他参数，排除已处理的dtype相关参数
        model_kwargs.update({k: v for k, v in kwargs.items() if k not in ["torch_dtype", "dtype"]})
        
        self._client = AutoModelForCausalLM.from_pretrained(self._model_id, **model_kwargs)
        
        # MUSA GPU特殊处理：手动移动模型到MUSA GPU
        if is_musa_gpu:
            self._client = self._client.to("musa")
            print("Model moved to MUSA GPU")
        
        # 设置 pad_token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        print(f"[QwenClient] Model loaded in {self._memory_mode} mode")
        return

    def _load_model(self) -> None:
        """Load model into memory."""
        if self._client is None:
            self._init_agent(**self._model_kwargs)
    
    def _unload_model(self) -> None:
        """Unload model from GPU memory."""
        if self._client is not None:
            print("[QwenClient] Unloading model from GPU memory...")
            self._client = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self._is_musa_gpu:
                torch.musa.empty_cache()
            print("[QwenClient] Model unloaded, memory freed")
    
    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> str:
        # Load model if not already loaded
        self._load_model()
        
        llm_config.pop("model", None)
        # temperature must be positive, 1e-5 works same as 0
        llm_config["temperature"] = max(llm_config.get("temperature", 1e-5), 1e-5)

        response = None
        num_attempt: int = 0
        while num_attempt < self._max_attempt:
            try:
                # 将消息转换为 Qwen 格式
                formatted_messages = self._format_messages_for_qwen(messages)
                
                # 使用 tokenizer 的 apply_chat_template 方法
                if hasattr(self._tokenizer, 'apply_chat_template'):
                    input_ids = self._tokenizer.apply_chat_template(
                        formatted_messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                else:
                    # 如果没有 apply_chat_template，手动构建输入
                    text = self._manual_format_messages(messages)
                    input_ids = self._tokenizer.encode(text, return_tensors="pt")
                
                # 移动到正确的设备
                device = next(self._client.parameters()).device
                input_ids = input_ids.to(device)

                # 生成参数
                generation_kwargs = {
                    "pad_token_id": self._tokenizer.eos_token_id,
                    "do_sample": llm_config.get("temperature", 1e-5) > 1e-5,
                    "temperature": llm_config.get("temperature", 1e-5),
                    "max_new_tokens": llm_config.get("max_new_tokens", 1024),
                }
                
                # 移除已处理的参数
                for key in ["temperature", "max_new_tokens"]:
                    llm_config.pop(key, None)
                
                # 添加其他生成参数
                generation_kwargs.update(llm_config)

                outputs = self._client.generate(input_ids, **generation_kwargs)

                # 提取新生成的部分
                response = outputs[0][input_ids.shape[-1]:]
                
                # Print the raw token IDs for debugging
                print(f"\n[QwenClient] Generated token count: {len(response)} tokens")

                break

            except Exception as e:
                self.warning(f"  Failed due to Exception: {e}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning("  Retrying...")
        
        # Unload model if in unload_after_use mode
        if self._memory_mode == "unload_after_use":
            self._unload_model()

        return response

    def _format_messages_for_qwen(self, messages: List[dict]) -> List[dict]:
        """将消息格式化为 Qwen 模型期望的格式"""
        formatted_messages = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Qwen 模型通常使用 "user", "assistant", "system" 角色
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
                
            formatted_messages.append({"role": role, "content": content})
        
        return formatted_messages

    def _manual_format_messages(self, messages: List[dict]) -> str:
        """手动格式化消息（当没有 apply_chat_template 时使用）"""
        formatted_text = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_text += f"System: {content}\n"
            elif role in ["user", "human"]:
                formatted_text += f"Human: {content}\n"
            elif role in ["assistant", "ai"]:
                formatted_text += f"Assistant: {content}\n"
        
        formatted_text += "Assistant: "
        return formatted_text

    def _get_content_from_response(self, response, messages: List[dict] = None) -> str:
        try:
            content = self._tokenizer.decode(response, skip_special_tokens=True)
            
            # Print decoded content
            print(f"\n[QwenClient] Decoded content:")
            print(f"{'-'*80}")
            print(content)
            print(f"{'-'*80}")
            
            if content is None:
                warning_message = "Non-Content returned"

                self.warning(warning_message)
                self.debug(f"  -- Complete response: {response}")
                if messages is not None and len(messages) >= 1:
                    self.debug(f"  -- Last message: {messages[-1]}")

                content = ""
        except Exception:
            content = ""

        return content
    
    def close(self):
        """Close the active memory, connections, and unload model."""
        # Unload model before closing
        self._unload_model()
        super().close()
