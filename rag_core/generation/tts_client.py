"""
TTS 客户端 - CosyVoice 语音合成
支持流式和完整音频生成
"""

import requests
import os
from typing import Optional, Generator
from rag_core.utils.logger import logger


class TTSClient:
    """CosyVoice TTS 客户端"""

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or os.getenv(
            "TTS_SERVER", "http://172.22.11.92:9880"
        )
        self.sample_rate = self._get_sample_rate()
        logger.info(f"[TTS] 服务地址: {self.server_url}, 采样率: {self.sample_rate}")

    def _get_sample_rate(self) -> int:
        """获取TTS服务采样率"""
        try:
            resp = requests.get(f"{self.server_url}/sample_rate", timeout=5)
            return resp.json()["sample_rate"]
        except Exception as e:
            logger.warning(f"[TTS] 无法获取采样率: {e}, 使用默认 22050")
            return 22050

    def generate_audio(
        self, text: str, instruct: Optional[str] = None
    ) -> Optional[bytes]:
        """
        生成完整音频（WAV格式）

        Args:
            text: 要合成的文本
            instruct: 语气指令，如 "用开心的语气说这句话"

        Returns:
            WAV格式音频数据（包含header），失败返回None
        """
        try:
            payload = {"text": text}
            if instruct:
                payload["instruct"] = instruct

            resp = requests.post(
                f"{self.server_url}/tts/complete", json=payload, timeout=60
            )
            resp.raise_for_status()

            audio_data = resp.content
            logger.info(f"[TTS] 生成成功: {len(audio_data)} bytes, 文本长度: {len(text)}")
            return audio_data

        except requests.exceptions.Timeout:
            logger.warning(f"[TTS] 请求超时: {text[:50]}...")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"[TTS] 连接失败，请确认服务是否启动: {self.server_url}")
            return None
        except Exception as e:
            logger.error(f"[TTS] 生成失败: {e}")
            return None

    def generate_stream(
        self, text: str, instruct: Optional[str] = None
    ) -> Optional[Generator[bytes, None, None]]:
        """
        生成流式音频（用于实时播放）

        Args:
            text: 要合成的文本
            instruct: 语气指令

        Yields:
            音频数据块
        """
        try:
            payload = {"text": text}
            if instruct:
                payload["instruct"] = instruct

            resp = requests.post(
                f"{self.server_url}/tts/complete",
                json=payload,
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()

            # 跳过WAV header (44 bytes)
            header_skipped = 0
            WAV_HEADER_SIZE = 44

            for chunk in resp.iter_content(chunk_size=4096):
                if not chunk:
                    continue

                if header_skipped < WAV_HEADER_SIZE:
                    skip = min(len(chunk), WAV_HEADER_SIZE - header_skipped)
                    chunk = chunk[skip:]
                    header_skipped += skip
                    if not chunk:
                        continue

                yield chunk

        except Exception as e:
            logger.error(f"[TTS] 流式生成失败: {e}")
            return None

    def test_connection(self) -> bool:
        """测试TTS服务连接"""
        try:
            resp = requests.get(f"{self.server_url}/sample_rate", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
