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
from rag_core.utils.session_manager import session_manager, MessageQueue

# 消息队列实例
message_queue = MessageQueue(max_queue_size=10, processing_timeout=60)

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


@app.get("/health")
async def health_check():
    """健康检查端点 - 检查核心服务状态"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {}
    }
    
    # 1. 检查 LLM 连接
    try:
        from rag_core.llm.llm_client import LLMClient
        client = LLMClient.get_instance()
        # 简单检查 client 是否可初始化
        health_status["services"]["llm"] = "ok"
    except Exception as e:
        health_status["services"]["llm"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # 2. 检查向量数据库
    try:
        from rag_core.knowledge.rag_tools import get_fact_indexer
        idx = get_fact_indexer()
        count = idx.count()
        health_status["services"]["vector_db"] = f"ok (chunks: {count})"
    except Exception as e:
        health_status["services"]["vector_db"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # 3. 检查 TTS 服务
    if TTS_ENABLED:
        if tts_client is not None:
            health_status["services"]["tts"] = "ok"
        else:
            health_status["services"]["tts"] = "not initialized"
    else:
        health_status["services"]["tts"] = "disabled"
    
    # 4. 检查 Qdrant 连接
    try:
        from rag_core.knowledge.indexing.fact_indexer import get_qdrant_client
        qdrant = get_qdrant_client()
        # 尝试获取 collections
        collections = qdrant.get_collections()
        health_status["services"]["qdrant"] = f"ok (collections: {len(collections.collections)})"
    except Exception as e:
        health_status["services"]["qdrant"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@app.get("/ready")
async def readiness_check():
    """就绪检查端点 - 检查服务是否可接收请求"""
    ready = True
    not_ready_reasons = []
    
    # 检查 LLM 是否可用
    try:
        from rag_core.llm.llm_client import LLMClient
        client = LLMClient.get_instance()
    except Exception as e:
        ready = False
        not_ready_reasons.append(f"LLM: {str(e)}")
    
    # 检查向量数据库是否有数据
    try:
        from rag_core.knowledge.rag_tools import get_fact_indexer
        idx = get_fact_indexer()
        if idx.count() == 0:
            ready = False
            not_ready_reasons.append("Vector DB empty - not indexed")
    except Exception as e:
        ready = False
        not_ready_reasons.append(f"Vector DB: {str(e)}")
    
    return {
        "ready": ready,
        "reasons": not_ready_reasons if not ready else ["all systems ready"]
    }


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
async def ws_chat(websocket: WebSocket, user_id: str = None):
    await websocket.accept()

    # 为每个连接分配独立的 session 和 agent（启用统一生成）
    # 使用传入的 user_id 或生成临时ID
    session_id = session_manager.create_session(user_id=user_id)
    agent = session_manager.get_agent(session_id)
    if hasattr(agent, "initialize"):
        await agent.initialize()

    logger.info(f"[WS] 新连接: {session_id}, user_id: {user_id or session_id}, 当前活跃会话: {len(session_manager.sessions)}")

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

            # 使用消息队列管理连续消息
            message_id = None
            try:
                message_id = message_queue.enqueue(session_id, user_text)
            except RuntimeError as e:
                logger.warning(f"[WS] 队列已满: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "text": "请求过于频繁，请稍后再试"
                }, ensure_ascii=False))
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

                # 标记消息处理完成
                if message_id:
                    message_queue.mark_completed(session_id, message_id, text)

                # 2. 🎤 流式生成并发送TTS音频
                if tts_streamer:
                    loop = asyncio.get_running_loop()
                    async def websocket_sender(data):
                        await websocket.send_text(json.dumps(data, ensure_ascii=False))

                    await tts_streamer.stream_audio(text, instruct, websocket_sender, loop)

            except Exception as e:
                logger.exception("WebSocket 消息处理异常")
                # 标记消息处理失败
                if message_id:
                    message_queue.mark_failed(session_id, message_id, str(e))

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
