# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
        max_attempt: int = 5, exponential_backoff_factor: int = None, unit_wait_time: int = 60, **kwargs,
    ) -> None:
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)
        assert "model" in llm_config, "`model` should be provided in `llm_config` to initialize `QwenClient`!"
        self._model_id: str = llm_config["model"]
        self._init_agent(**kwargs)

    def _init_agent(self, **kwargs) -> None:
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)

        # 设置模型参数
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if device != "cpu" else None,
        }
        # 简化 dtype 处理
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs.pop("dtype")
        elif "torch_dtype" in kwargs:
            dtype = kwargs.pop("torch_dtype")
        if dtype:
            if isinstance(dtype, str):
                dtype_str = dtype.replace("torch.", "").lower().strip()
                if hasattr(torch, dtype_str):
                    model_kwargs["dtype"] = getattr(torch, dtype_str)
                else:
                    model_kwargs["dtype"] = torch.float16
            else:
                model_kwargs["dtype"] = dtype
        else:
            model_kwargs["dtype"] = torch.float16
            # Add any other kwargs except dtype keys
            for k, v in kwargs.items():
                if k not in ["torch_dtype", "dtype"]:
                    model_kwargs[k] = v
            self._client = AutoModelForCausalLM.from_pretrained(self._model_id, **model_kwargs)
            # 设置 pad_token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> str:
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

                break

            except Exception as e:
                self.warning(f"  Failed due to Exception: {e}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning("  Retrying...")

        return response

    def _format_messages_for_qwen(self, messages: List[dict]) -> List[dict]:
        # 将消息格式化为 Qwen 模型期望的格式
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
        # 手动格式化消息（当没有 apply_chat_template 时使用）
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
