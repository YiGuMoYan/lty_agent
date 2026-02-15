import os
import asyncio
from enum import Enum
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI
import config
from rag_core.utils.logger import logger

class LLMErrorType(Enum):
    """LLM 错误类型分类"""
    TIMEOUT = "timeout"           # 超时
    RATE_LIMIT = "rate_limit"     # 速率限制
    API_ERROR = "api_error"       # API 错误
    PARSE_ERROR = "parse_error"   # 解析错误
    UNKNOWN = "unknown"           # 未知错误

class LLMError(Exception):
    """LLM 错误异常"""
    def __init__(self, message: str, error_type: LLMErrorType, is_retryable: bool = True):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.is_retryable = is_retryable

class LLMClient:
    """
    LLM 客户端 - 单例模式
    支持重试机制、错误分类、连接复用
    """
    _instance: Optional['LLMClient'] = None
    _initialized: bool = False

    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAYS = [2, 4, 8]  # 指数退避（秒）

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 避免重复初始化
        if LLMClient._initialized:
            return

        self.model_name = config.CHAT_MODEL_NAME
        self.base_url = config.CHAT_API_BASE
        self.api_key = config.CHAT_API_KEY

        # 可配置参数
        self.temperature = getattr(config, 'LLM_TEMPERATURE', 0.7)
        self.max_tokens = getattr(config, 'LLM_MAX_TOKENS', 2048)
        self.timeout = getattr(config, 'LLM_TIMEOUT', 120)

        logger.info(f"[LLMClient] Connecting to {self.base_url} (Model: {self.model_name})")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0  # 自定义重试机制
        )

        LLMClient._initialized = True

    @classmethod
    def get_instance(cls) -> 'LLMClient':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _classify_error(self, error: Exception) -> LLMErrorType:
        """错误分类"""
        error_msg = str(error).lower()

        if "timeout" in error_msg or "timed out" in error_msg:
            return LLMErrorType.TIMEOUT
        elif "rate limit" in error_msg or "429" in error_msg:
            return LLMErrorType.RATE_LIMIT
        elif "api" in error_msg or "500" in error_msg or "502" in error_msg or "503" in error_msg:
            return LLMErrorType.API_ERROR
        elif "json" in error_msg or "parse" in error_msg:
            return LLMErrorType.PARSE_ERROR
        else:
            return LLMErrorType.UNKNOWN

    def _is_retryable(self, error_type: LLMErrorType) -> bool:
        """判断错误是否可重试"""
        return error_type in [
            LLMErrorType.TIMEOUT,
            LLMErrorType.RATE_LIMIT,
            LLMErrorType.API_ERROR
        ]

    async def _retry_request(self, request_func, *args, **kwargs):
        """带重试的请求执行"""
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                return await request_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)

                logger.warning(f"[LLMClient] Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")

                # 不可重试的错误直接抛出
                if not self._is_retryable(error_type):
                    logger.warning(f"[LLMClient] Non-retryable error, giving up: {error_type.value}")
                    break

                # 达到最大重试次数
                if attempt >= self.MAX_RETRIES - 1:
                    break

                # 指数退避等待
                delay = self.RETRY_DELAYS[attempt] if attempt < len(self.RETRY_DELAYS) else self.RETRY_DELAYS[-1]
                logger.debug(f"[LLMClient] Retrying in {delay}s...")
                await asyncio.sleep(delay)

        # 所有重试都失败
        raise LLMError(
            f"Request failed after {self.MAX_RETRIES} attempts: {last_error}",
            self._classify_error(last_error) if last_error else LLMErrorType.UNKNOWN,
            is_retryable=False
        )

    async def chat_with_tools(self, messages, tools=None, tool_choice="auto"):
        """
        Chat completion with optional tool calling (Async).
        支持重试机制
        """
        async def _make_request():
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice=tool_choice if tools else None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message
            except Exception as e:
                # 包装异常
                error_type = self._classify_error(e)
                raise LLMError(f"Chat completion failed: {e}", error_type)

        try:
            return await self._retry_request(_make_request)
        except LLMError as e:
            logger.error(f"[LLMClient] Final error: {e.message}")
            return None

    async def chat(self, messages, temperature=None):
        """
        简单聊天接口（无工具调用）
        """
        async def _make_request():
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                error_type = self._classify_error(e)
                raise LLMError(f"Chat failed: {e}", error_type)

        try:
            result = await self._retry_request(_make_request)
            return result if result else "（数据流中断...）"
        except LLMError as e:
            logger.error(f"[LLMClient] Final error: {e.message}")
            return "（服务暂时不可用，请稍后再试...）"
