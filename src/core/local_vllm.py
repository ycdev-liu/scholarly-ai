from typing import Optional, List
import requests
from pydantic import BaseModel, Field
from langchain.llms.base import LLM

class LocalQwenLLM(LLM, BaseModel):
    api_url: str
    model_id: str

    class Config:
        # 允许额外字段，避免 Pydantic 校验报错
        extra = "allow"

    @property
    def _llm_type(self) -> str:
        return "local_qwen"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices or "message" not in choices[0]:
            raise ValueError(f"Unexpected response: {data}")
        return choices[0]["message"]["content"]

# 初始化自定义 LLM
local_llm = LocalQwenLLM(
    api_url="http://192.168.1.102:8000/v1/chat/completions",
    model_id="/root/large_model_project/models/Qwen2.5-3B-Instruct"
)


result = local_llm("你好，请介绍一下自己。")
print(result)
