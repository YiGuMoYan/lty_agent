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
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from config import WS_PORT, BASE_DIR, TTS_ENABLED
from rag_core.companion_agent import CompanionAgent
from rag_core.tts_client import TTSClient

app = FastAPI()

# 静态文件：Live2D 模型资源
app.mount("/live2d", StaticFiles(directory=os.path.join(BASE_DIR, "live2d")), name="live2d")

# 静态文件：HTML 页面（直接挂载项目根目录下的 html）
# 用单独路由返回 HTML，避免挂载整个根目录

# 多用户隔离：每个连接独立的 agent 实例
active_agents: Dict[str, CompanionAgent] = {}

# TTS 客户端（全局单例）
tts_client = None
if TTS_ENABLED:
    try:
        tts_client = TTSClient()
        if tts_client.test_connection():
            print(f"[TTS] ✓ 服务连接成功")
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
    session_id = str(uuid.uuid4())
    agent = CompanionAgent(use_emotional_mode=True, use_unified_generator=True)
    active_agents[session_id] = agent
    print(f"[WS] 新连接: {session_id}, 当前活跃: {len(active_agents)}")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            user_text = msg.get("text", "").strip()
            if not user_text:
                continue

            try:
                # 🚀 统一生成模式：一次LLM调用同时生成对话和Live2D
                # UPDATE: agent methods are now async, so we await them directly!
                # text, instruct, emotion_state, live2d = await loop.run_in_executor(
                #    None, agent.chat_with_live2d_unified, user_text
                # )
                text, instruct, emotion_state, live2d = await agent.chat_with_live2d_unified(user_text)

                # 准备文本响应数据（暂时不发，等音频准备好）
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

                # 标记文本是否已发送
                text_sent = False

                # 🎤 流式生成并发送TTS音频
                if tts_client:
                    try:
                        # 1. 获取 TTS 流生成器
                        # Note: TTS client uses requests (sync), so we still need run_in_executor for the stream generation
                        # or refactor TTSClient to be async (future work). For now, keep it in executor.
                        loop = asyncio.get_running_loop()

                        def get_tts_stream():
                            return tts_client.generate_stream(text, instruct)

                        # 安全的 next 函数，避免 StopIteration 传播到 Future
                        def safe_next(iterator):
                            try:
                                return next(iterator)
                            except StopIteration:
                                return None

                        stream_iterator = await loop.run_in_executor(None, get_tts_stream)

                        if stream_iterator:
                            # 转换为迭代器
                            iterator = iter(stream_iterator)
                            first_chunk = None

                            # 2. 预读第一个音频块 (关键：等待声音准备好)
                            # 这一步会阻塞等待 TTS 首包，确保音画同步
                            try:
                                first_chunk = await loop.run_in_executor(None, safe_next, iterator)
                            except Exception as e:
                                print(f"[TTS] 首包获取失败: {e}")

                            # 3. 声音准备好了 (或确认无声音)，发送文本和动作
                            # 此时发送，用户看到的文字和听到的声音是同步的
                            await websocket.send_text(json.dumps(response_payload, ensure_ascii=False))
                            text_sent = True

                            # 4. 如果有音频，开始流式发送
                            if first_chunk:
                                # 发送音频开始标记
                                await websocket.send_text(json.dumps({
                                    "type": "audio_start",
                                    "sample_rate": tts_client.sample_rate
                                }))

                                chunk_count = 0

                                # 发送第一个块
                                chunk_count += 1
                                chunk_base64 = base64.b64encode(first_chunk).decode('utf-8')
                                await websocket.send_text(json.dumps({
                                    "type": "audio_chunk",
                                    "data": chunk_base64,
                                    "chunk_id": chunk_count
                                }))

                                # 5. 循环读取并发送剩余块
                                # 关键优化：在 executor 中读取下一个块，避免阻塞 asyncio 事件循环
                                while True:
                                    try:
                                        # 在线程池中读取，防止卡顿
                                        chunk = await loop.run_in_executor(None, safe_next, iterator)

                                        if chunk:
                                            chunk_count += 1
                                            chunk_base64 = base64.b64encode(chunk).decode('utf-8')
                                            await websocket.send_text(json.dumps({
                                                "type": "audio_chunk",
                                                "data": chunk_base64,
                                                "chunk_id": chunk_count
                                            }))
                                        else:
                                            # None 表示迭代结束
                                            break
                                    except Exception as e:
                                        print(f"[TTS] 流读取中断: {e}")
                                        break

                                print(f"[TTS] 流式音频发送完成: {chunk_count} chunks")

                                # 发送音频结束标记
                                await websocket.send_text(json.dumps({
                                    "type": "audio_end",
                                    "total_chunks": chunk_count
                                }))
                        else:
                            print("[TTS] 未获取到音频流")

                    except Exception as e:
                        print(f"[TTS] 流式音频处理失败: {e}")
                        traceback.print_exc()
                        # 发送错误通知前端（可选）

                # 兜底：如果上面因为 TTS 失败没发文本，这里补发
                if not text_sent:
                    await websocket.send_text(json.dumps(response_payload, ensure_ascii=False))

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
        # 清理会话
        if session_id in active_agents:
            del active_agents[session_id]
            print(f"[WS] 清理会话: {session_id}, 剩余: {len(active_agents)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=WS_PORT)
