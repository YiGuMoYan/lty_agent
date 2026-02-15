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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from config import WS_PORT, BASE_DIR, TTS_ENABLED
from rag_core.agent.companion_agent import CompanionAgent
from rag_core.generation.tts_client import TTSClient
from rag_core.generation.tts_streamer import TTSStreamer

app = FastAPI()

# 静态文件：Live2D 模型资源
app.mount("/live2d", StaticFiles(directory=os.path.join(BASE_DIR, "live2d")), name="live2d")

# 静态文件：HTML 页面（直接挂载项目根目录下的 html）
# 用单独路由返回 HTML，避免挂载整个根目录

class SessionManager:
    def __init__(self, ttl_seconds=1800):
        self.sessions: Dict[str, Dict] = {} # {session_id: {'agent': agent, 'last_active': timestamp}}
        self.ttl_seconds = ttl_seconds

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        agent = CompanionAgent(use_emotional_mode=True, use_unified_generator=True)
        self.sessions[session_id] = {
            'agent': agent,
            'last_active': time.time()
        }
        return session_id

    def get_agent(self, session_id: str) -> Optional[CompanionAgent]:
        if session_id in self.sessions:
            self.sessions[session_id]['last_active'] = time.time()
            return self.sessions[session_id]['agent']
        return None

    def remove_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def cleanup_expired(self):
        now = time.time()
        expired = [sid for sid, data in self.sessions.items() if now - data['last_active'] > self.ttl_seconds]
        for sid in expired:
            print(f"[SessionManager] Cleaning up expired session: {sid}")
            del self.sessions[sid]
        return len(expired)

session_manager = SessionManager()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_cleanup())

async def background_cleanup():
    while True:
        await asyncio.sleep(300) # Check every 5 mins
        count = session_manager.cleanup_expired()
        if count > 0:
            print(f"[Cleanup] Removed {count} expired sessions")

# TTS 服务初始化
tts_client = None
tts_streamer = None

if TTS_ENABLED:
    try:
        tts_client = TTSClient()
        if tts_client.test_connection():
            print(f"[TTS] ✓ 服务连接成功")
            tts_streamer = TTSStreamer(tts_client)
        else:
            print(f"[TTS] ⚠️  服务连接失败，TTS功能将禁用")
            tts_client = None
    except Exception as e:
        print(f"[TTS] 初始化失败: {e}")
        tts_client = None


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

    print(f"[WS] 新连接: {session_id}, 当前活跃会话: {len(session_manager.sessions)}")

    try:
        while True:
            data = await websocket.receive_text()

            # 刷新活跃时间
            agent = session_manager.get_agent(session_id)
            if not agent:
                # 理论上不会发生，除非被清理
                print(f"[WS] 会话已失效: {session_id}")
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
                traceback.print_exc()
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
        print(f"[WS] 客户端断开: {session_id}")
    except Exception as e:
        traceback.print_exc()
        print(f"[WS] 错误: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}, ensure_ascii=False))
        except Exception:
            pass
    finally:
        # 清理会话 (连接断开即清理，或者保留等待超时？)
        # 这里选择立即清理，如果需要重连机制则不应立即删除
        # 但考虑到目前没有重连恢复逻辑，保持原逻辑清理
        session_manager.remove_session(session_id)
        print(f"[WS] 清理会话: {session_id}, 剩余: {len(session_manager.sessions)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=WS_PORT)
