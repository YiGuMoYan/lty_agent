import asyncio
import base64
import json
import traceback
from typing import AsyncGenerator, Optional, Callable, Any, Dict

from rag_core.generation.async_tts_client import AsyncTTSClient
from rag_core.utils.logger import logger

class TTSStreamer:
    """
    Handles TTS streaming logic for WebSocket connections using AsyncTTSClient.
    """
    def __init__(self, tts_client: Optional[AsyncTTSClient] = None) -> None:
        self.tts_client = tts_client

    async def stream_audio(
        self,
        text: str,
        instruct: Optional[str],
        websocket_send_json: Callable[[Dict[str, Any]], Any],
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> bool:
        """
        Generates and sends audio chunks via websocket using async iterator.

        Args:
            text: Text to synthesize
            instruct: Emotion instruction
            websocket_send_json: Async function to send JSON to websocket
            loop: Deprecated, kept for compatibility but not used.

        Returns:
            bool: True if audio was successfully generated and sent, False otherwise.
        """
        if not self.tts_client:
            return False

        try:
            # 1. Start generation (Async Generator)
            stream_iterator = self.tts_client.generate_stream(text, instruct)

            # 2. Pre-fetch first chunk to ensure stream is working
            # We manually iterate to get the first chunk
            first_chunk = None
            async for chunk in stream_iterator:
                first_chunk = chunk
                break

            if not first_chunk:
                logger.warning("[TTS] Stream empty or failed to start")
                return False

            # 3. Send Audio Start
            await websocket_send_json({
                "type": "audio_start",
                "sample_rate": self.tts_client.sample_rate
            })

            chunk_count = 0

            # 4. Send First Chunk
            chunk_count += 1
            await self._send_chunk(first_chunk, chunk_count, websocket_send_json)

            # 5. Stream remaining chunks
            async for chunk in stream_iterator:
                if chunk:
                    chunk_count += 1
                    await self._send_chunk(chunk, chunk_count, websocket_send_json)

            logger.info(f"[TTS] Stream finished: {chunk_count} chunks")

            # 6. Send Audio End
            await websocket_send_json({
                "type": "audio_end",
                "total_chunks": chunk_count
            })

            return True

        except Exception as e:
            logger.error(f"[TTS] Streaming failed: {e}")
            traceback.print_exc()
            return False

    async def _send_chunk(
        self,
        chunk: bytes,
        chunk_id: int,
        send_func: Callable[[Dict[str, Any]], Any]
    ) -> None:
        chunk_base64 = base64.b64encode(chunk).decode('utf-8')
        await send_func({
            "type": "audio_chunk",
            "data": chunk_base64,
            "chunk_id": chunk_id
        })
