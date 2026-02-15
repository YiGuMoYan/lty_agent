"""
WebSocket 聊天服务 — FastAPI
将 CompanionAgent 的情感回复与 Live2D 参数打通
"""

import asyncio
import json
import os
import traceback
import uuid
import base64
import time
from typing import Dict, Optional
from rag_core.utils.logger import logger

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from config import WS_PORT, BASE_DIR, TTS_ENABLED
from rag_core.agent.companion_agent import CompanionAgent
from rag_core.generation.async_tts_client import AsyncTTSClient
from rag_core.generation.tts_streamer import TTSStreamer
from rag_core.utils.session_manager import session_manager

app = FastAPI()

# ... (omitted code) ...

# TTS 服务初始化
tts_client = None
tts_streamer = None

@app.on_event("startup")
async def startup_event():
    global tts_client, tts_streamer

    # Start cleanup task
    asyncio.create_task(background_cleanup())

    # Initialize TTS
    if TTS_ENABLED:
        try:
            tts_client = AsyncTTSClient()
            await tts_client.initialize()
            logger.info(f"[TTS] 服务连接成功 (Rate: {tts_client.sample_rate})")
            tts_streamer = TTSStreamer(tts_client)
        except Exception as e:
            logger.error(f"[TTS] 初始化失败: {e}")
            tts_client = None

@app.on_event("shutdown")
async def shutdown_event():
    if tts_client:
        await tts_client.close()

# ... (rest of the file)


@app.get("/")
async def root():
    return RedirectResponse(url="/viewer")


@app.get("/viewer")
async def viewer():
    html_path = os.path.join(BASE_DIR, "live2d_viewer.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=content)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()

    # 为每个连接分配独立的 session 和 agent（启用统一生成）
    session_id = session_manager.create_session()
    agent = session_manager.get_agent(session_id)
    if hasattr(agent, "initialize"):
        await agent.initialize()

    logger.info(f"[WS] 新连接: {session_id}, 当前活跃会话: {len(session_manager.sessions)}")

    try:
        while True:
            data = await websocket.receive_text()

            # 刷新活跃时间
            agent = session_manager.get_agent(session_id)
            if not agent:
                # 理论上不会发生，除非被清理
                logger.warning(f"[WS] 会话已失效: {session_id}")
                break

            msg = json.loads(data)
            user_text = msg.get("text", "").strip()
            if not user_text:
                continue

            try:
                # 🚀 统一生成模式：一次LLM调用同时生成对话和Live2D
                text, instruct, emotion_state, live2d = await agent.chat_with_live2d_unified(user_text)

                # 准备文本响应数据
                response_payload = {
                    "type": "response",
                    "text": text,
                    "instruct": instruct,
                    "emotion": emotion_state.primary_emotion,
                    "intensity": round(emotion_state.intensity, 2),
                    "live2d_params": live2d["params"],
                    "pose": live2d.get("pose"),
                    "action_sequence": live2d.get("action_sequence", []),
                }

                # 1. 优先发送文本和动作（优化首字/首帧延迟）
                await websocket.send_text(json.dumps(response_payload, ensure_ascii=False))

                # 2. 🎤 流式生成并发送TTS音频
                if tts_streamer:
                    loop = asyncio.get_running_loop()
                    async def websocket_sender(data):
                        await websocket.send_text(json.dumps(data, ensure_ascii=False))

                    await tts_streamer.stream_audio(text, instruct, websocket_sender, loop)

            except Exception as e:
                logger.exception("WebSocket 消息处理异常")
                err_resp = {
                    "type": "error",
                    "text": f"出错了: {e}",
                    "emotion": "平静",
                    "intensity": 0.3,
                    "live2d_params": {},
                    "pose": None,
                    "action_sequence": [],
                    "instruct": ""
                }
                await websocket.send_text(json.dumps(err_resp, ensure_ascii=False))

    except WebSocketDisconnect:
        logger.info(f"[WS] 客户端断开: {session_id}")
    except Exception as e:
        logger.exception(f"[WS] 错误: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}, ensure_ascii=False))
        except Exception:
            pass
    finally:
        # 清理会话 (连接断开即清理，或者保留等待超时？)
        # 这里选择立即清理，如果需要重连机制则不应立即删除
        # 但考虑到目前没有重连恢复逻辑，保持原逻辑清理
        session_manager.remove_session(session_id)
        logger.info(f"[WS] 清理会话: {session_id}, 剩余: {len(session_manager.sessions)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=WS_PORT)
