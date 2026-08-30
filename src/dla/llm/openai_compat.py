"""LLM 接入层（doc/01 §3 / D23）。

提供两个 ``LLMClient`` 实现：
- ``HTTPLLMClient``：OpenAI 兼容 HTTP（标准库 ``urllib``，零第三方依赖），适配任意
  OpenAI / DeepSeek / 通义 / 本地网关，切换只改环境变量。
- ``FakeLLMClient``：离线实现，无需 API key 即可跑通完整闭环（测试 / 演示）。通过
  ``response_format`` 是否含 json 来区分「分析请求」与「对话请求」。

工厂 ``make_llm_client(settings)``：当 ``api_key`` 为空或占位符时自动退回 Fake，
否则用真实 HTTP 客户端——保证无 key 也能开发/测试。
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from ..core.errors import LlmError, LlmRateLimitError, LlmSchemaError, LlmTimeoutError
from ..core.ports import ChatMessage, LlmResult

# 离线词典：FakeLLM 用于模拟「LLM 抽取」的最小信号
_FAKE_MOOD = {
    "难": ("user_mood.frustrated", 0.8),
    "不会": ("user_mood.frustrated", 0.7),
    "烦": ("user_mood.frustrated", 0.75),
    "挫败": ("user_mood.frustrated", 0.85),
    "急": ("user_temper.impatient", 0.7),
    "快点": ("user_temper.impatient", 0.6),
    "慢": ("user_temper.impatient", 0.5),
    "开心": ("user_mood.excited", 0.8),
    "喜欢": ("user_mood.excited", 0.7),
    "棒": ("user_mood.excited", 0.75),
    "紧张": ("user_mood.anxious", 0.7),
    "怕": ("user_mood.anxious", 0.65),
    "完美": ("user_temper.perfectionist", 0.6),
}


def _is_json_request(response_format: Optional[dict]) -> bool:
    if not response_format:
        return False
    return "json" in str(response_format.get("type", "")).lower() or "json" in str(
        response_format.get("response_schema", "")
    ).lower()


class HTTPLLMClient:
    """OpenAI 兼容 HTTP 客户端（零依赖）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: Optional[dict] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> LlmResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]
                content = choice.get("message", {}).get("content", "")
                tool_calls = choice.get("message", {}).get("tool_calls")
                usage = data.get("usage")
                return LlmResult(content=content, raw=data, tool_calls=tool_calls, usage=usage)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    last_err = LlmRateLimitError(f"限流: {e}")
                else:
                    last_err = LlmError(f"HTTP {e.code}: {e}")
                time.sleep(min(2**attempt, 8))
            except urllib.error.URLError as e:
                last_err = LlmTimeoutError(f"网络错误: {e}")
                time.sleep(min(2**attempt, 8))
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                last_err = LlmSchemaError(f"响应解析失败: {e}")
                break
        assert last_err is not None
        raise last_err


class FakeLLMClient:
    """离线假 LLM：无 key 即可跑通闭环。

    - 分析请求（response_format 含 json）→ 基于内置词典产出结构化 JSON。
    - 对话请求 → 生成风格化文本回复，并附上简单 turn_summary。
    """

    def __init__(self, model: str = "fake-llm") -> None:
        self.model = model

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: Optional[dict] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> LlmResult:
        last_user = ""
        for m in reversed(messages):
            if m.role == "user":
                last_user = m.content
                break

        if _is_json_request(response_format):
            return self._fake_analyze(last_user)
        return self._fake_chat(last_user, messages)

    def _fake_analyze(self, text: str) -> LlmResult:
        # analyze() 传入的是「上一轮 agent 说：...\n用户现在说：...」组合串，
        # 只分析用户部分（与真实 LLM 受系统提示约束「仅从用户言行提炼」一致）。
        marker = "用户现在说："
        user_part = text.split(marker, 1)[1] if marker in text else text
        extractions: List[dict] = []
        seen: set[str] = set()
        for token, (key, intensity) in _FAKE_MOOD.items():
            if token in user_part and key not in seen:
                extractions.append({"key": key, "intensity": intensity, "source": "llm"})
                seen.add(key)
        summary = (user_part[:90] + "…") if len(user_part) > 90 else user_part
        result = {
            "extractions": extractions,
            "scene_ops": [],
            "agent_ops": [],
            "turn_summary": f"用户表达了相关情绪/态度：{summary}" if extractions else f"用户：{summary}",
            "confidence": 0.7 if extractions else 0.5,
        }
        return LlmResult(content=json.dumps(result, ensure_ascii=False))

    def _fake_chat(self, text: str, messages: Sequence[ChatMessage]) -> LlmResult:
        # 根据 system 中是否出现「鼓励」/「客服」做风格化（仅演示用）
        system_hint = ""
        for m in messages:
            if m.role == "system":
                system_hint = m.content
                break
        # 引用用户原话片段，保证不同轮次的回复文本有差异，避免离线时触发重复护栏误判
        snippet = text[:8] if text else "刚才"
        if "鼓励" in system_hint or "encourag" in system_hint.lower():
            reply = f"别急，关于「{snippet}」我们一步步来，你已经做得很好了～"
        elif "客服" in system_hint:
            reply = f"您好，很高兴为您服务，关于「{snippet}」有什么可以帮您？"
        else:
            reply = f"我明白你的意思（{snippet}），我们可以继续往下聊。"
        summary = (text[:90] + "…") if len(text) > 90 else text
        return LlmResult(content=f"{reply}\n\n<turn_summary>{summary}</turn_summary>")


def make_llm_client(
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int = 60,
    max_retries: int = 2,
) -> "LLMClientLike":
    """工厂：key 为空/占位符 → FakeLLM（离线）；否则 HTTP。"""
    placeholder = api_key.strip() in ("", "sk-placeholder", "placeholder")
    if placeholder:
        return FakeLLMClient(model=model)
    return HTTPLLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


class LLMClientLike:
    """结构化类型提示（运行期协议）。"""
