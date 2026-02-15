import aiohttp
import os
import asyncio
from typing import Optional, AsyncGenerator
from rag_core.utils.logger import logger

class AsyncTTSClient:
    """
    Asynchronous CosyVoice TTS Client using aiohttp.
    Supports both complete audio generation and streaming.
    """

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or os.getenv(
            "TTS_SERVER", "http://172.22.11.92:9880"
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.sample_rate = 22050  # Default fallback
        self._initialized = False

    async def initialize(self):
        """Initialize the client session and fetch sample rate."""
        if self._initialized:
            return

        self.session = aiohttp.ClientSession()
        self.sample_rate = await self._get_sample_rate()
        logger.info(f"[AsyncTTS] Initialized. URL: {self.server_url}, Sample Rate: {self.sample_rate}")
        self._initialized = True

    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
        self._initialized = False

    async def _get_sample_rate(self) -> int:
        """Fetch sample rate from TTS server."""
        if not self.session:
            return 22050

        try:
            async with self.session.get(f"{self.server_url}/sample_rate", timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("sample_rate", 22050)
        except Exception as e:
            logger.warning(f"[AsyncTTS] Failed to get sample rate: {e}")
        return 22050

    async def generate_audio(self, text: str, instruct: Optional[str] = None) -> Optional[bytes]:
        """
        Generate complete audio (WAV format).

        Args:
            text: Text to synthesize
            instruct: Emotion instruction

        Returns:
            WAV audio bytes or None if failed
        """
        if not self.session:
            await self.initialize()

        payload = {"text": text}
        if instruct:
            payload["instruct"] = instruct

        try:
            async with self.session.post(
                f"{self.server_url}/tts/complete",
                json=payload,
                timeout=60
            ) as resp:
                if resp.status != 200:
                    logger.error(f"[AsyncTTS] Error {resp.status}: {await resp.text()}")
                    return None

                data = await resp.read()
                logger.debug(f"[AsyncTTS] Generated {len(data)} bytes for: {text[:20]}...")
                return data
        except Exception as e:
            logger.error(f"[AsyncTTS] Generate failed: {e}")
            return None

    async def generate_stream(self, text: str, instruct: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        """
        Generate streaming audio.

        Yields:
            Audio chunks (bytes)
        """
        if not self.session:
            await self.initialize()

        payload = {"text": text}
        if instruct:
            payload["instruct"] = instruct

        try:
            async with self.session.post(
                f"{self.server_url}/tts/complete",
                json=payload,
                timeout=300  # Longer timeout for streaming
            ) as resp:
                if resp.status != 200:
                    logger.error(f"[AsyncTTS] Stream Error {resp.status}")
                    return

                # Skip WAV header (44 bytes) logic usually handled by consumer or here
                # For consistency with previous implementation, we yield raw chunks
                # and let consumer handle header skipping if needed,
                # BUT previous implementation skipped header in client.
                # Let's handle header skipping here for the generator.

                header_skipped = 0
                WAV_HEADER_SIZE = 44

                async for chunk in resp.content.iter_chunked(4096):
                    if not chunk:
                        continue

                    if header_skipped < WAV_HEADER_SIZE:
                        skip = min(len(chunk), WAV_HEADER_SIZE - header_skipped)
                        data_to_yield = chunk[skip:]
                        header_skipped += skip

                        if data_to_yield:
                            yield data_to_yield
                    else:
                        yield chunk

        except Exception as e:
            logger.error(f"[AsyncTTS] Stream failed: {e}")
