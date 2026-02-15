import asyncio
import base64
import json
import traceback
from typing import AsyncGenerator, Optional, Callable, Any

class TTSStreamer:
    """
    Handles TTS streaming logic for WebSocket connections.
    Encapsulates the complexity of thread pool execution, chunking, and base64 encoding.
    """
    def __init__(self, tts_client: Any):
        self.tts_client = tts_client

    async def stream_audio(self,
                          text: str,
                          instruct: str,
                          websocket_send_json: Callable[[dict], Any],
                          loop: asyncio.AbstractEventLoop) -> bool:
        """
        Generates and sends audio chunks via websocket.

        Args:
            text: Text to synthesize
            instruct: Emotion instruction
            websocket_send_json: Async function to send JSON to websocket
            loop: Current event loop

        Returns:
            bool: True if audio was successfully generated and sent, False otherwise.
        """
        if not self.tts_client:
            return False

        try:
            # 1. Define generator function (sync)
            def get_tts_stream():
                return self.tts_client.generate_stream(text, instruct)

            # Helper for safe iteration in executor
            def safe_next(iterator):
                try:
                    return next(iterator)
                except StopIteration:
                    return None

            # 2. Start generation in thread pool
            stream_iterator = await loop.run_in_executor(None, get_tts_stream)

            if not stream_iterator:
                print("[TTS] No stream iterator returned")
                return False

            iterator = iter(stream_iterator)

            # 3. Pre-fetch first chunk (Blocking wait for first byte)
            try:
                first_chunk = await loop.run_in_executor(None, safe_next, iterator)
            except Exception as e:
                print(f"[TTS] Failed to get first chunk: {e}")
                return False

            if not first_chunk:
                print("[TTS] Stream empty")
                return False

            # 4. Send Audio Start
            await websocket_send_json({
                "type": "audio_start",
                "sample_rate": self.tts_client.sample_rate
            })

            chunk_count = 0

            # 5. Send First Chunk
            chunk_count += 1
            await self._send_chunk(first_chunk, chunk_count, websocket_send_json)

            # 6. Stream remaining chunks
            while True:
                try:
                    # Fetch next chunk in executor to avoid blocking event loop
                    chunk = await loop.run_in_executor(None, safe_next, iterator)

                    if chunk:
                        chunk_count += 1
                        await self._send_chunk(chunk, chunk_count, websocket_send_json)
                    else:
                        break # End of stream
                except Exception as e:
                    print(f"[TTS] Stream interruption: {e}")
                    break

            print(f"[TTS] Stream finished: {chunk_count} chunks")

            # 7. Send Audio End
            await websocket_send_json({
                "type": "audio_end",
                "total_chunks": chunk_count
            })

            return True

        except Exception as e:
            print(f"[TTS] Streaming failed: {e}")
            traceback.print_exc()
            return False

    async def _send_chunk(self, chunk: bytes, chunk_id: int, send_func: Callable):
        chunk_base64 = base64.b64encode(chunk).decode('utf-8')
        await send_func({
            "type": "audio_chunk",
            "data": chunk_base64,
            "chunk_id": chunk_id
        })
