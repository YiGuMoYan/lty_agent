# -*- coding: utf-8 -*-
"""
洛天依语音对话客户端
RAG 生成文本回复 → CosyVoice 流式合成语音 → 实时播放
"""

import sys
import os
import io
import argparse
import threading
import queue
import requests

if sys.platform.startswith("win"):
    os.system("chcp 65001 >nul")

try:
    import pyaudio

    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

from rag_core.companion_agent import CompanionAgent

# CosyVoice 服务地址
TTS_SERVER = os.getenv("TTS_SERVER", "http://172.22.11.92:9880")

# 音频参数
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16
CHUNK_SIZE = 4096  # 每次读取字节数
WAV_HEADER_SIZE = 44
PREBUFFER_BYTES = 24000 * 2 * 2  # 约2秒缓冲


def get_sample_rate():
    """从 TTS 服务获取采样率"""
    try:
        resp = requests.get(f"{TTS_SERVER}/sample_rate", timeout=5)
        return resp.json()["sample_rate"]
    except Exception:
        return 22050  # fallback


def tts_stream(text: str, instruct: str = None):
    """请求 TTS 服务，返回流式音频的 response 对象（带 wav header）"""
    payload = {"text": text}
    if instruct:
        payload["instruct"] = instruct
    resp = requests.post(
        f"{TTS_SERVER}/tts/complete",
        json=payload,
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()
    return resp


def play_stream(resp, sample_rate):
    """实时播放流式音频（带缓冲，跳过 wav header）"""
    if not HAS_PYAUDIO:
        print("  [未安装 pyaudio，音频将保存到 output.wav]")
        data = resp.content
        with open("output.wav", "wb") as f:
            f.write(data)
        print("  [已保存 output.wav]")
        return

    buf = queue.Queue()
    sentinel = object()
    prebuffer = bytearray()
    prebuffer_done = threading.Event()
    header_skipped = 0

    def recv_thread():
        nonlocal header_skipped
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            if header_skipped < WAV_HEADER_SIZE:
                skip = min(len(chunk), WAV_HEADER_SIZE - header_skipped)
                chunk = chunk[skip:]
                header_skipped += skip
                if not chunk:
                    continue
            if not prebuffer_done.is_set():
                prebuffer.extend(chunk)
                if len(prebuffer) >= PREBUFFER_BYTES:
                    buf.put(bytes(prebuffer))
                    prebuffer_done.set()
            else:
                buf.put(chunk)
        if not prebuffer_done.is_set() and prebuffer:
            buf.put(bytes(prebuffer))
            prebuffer_done.set()
        buf.put(sentinel)

    t = threading.Thread(target=recv_thread, daemon=True)
    t.start()

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=sample_rate,
        output=True,
    )

    prebuffer_done.wait()

    try:
        while True:
            data = buf.get()
            if data is sentinel:
                break
            stream.write(data)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        t.join()


def chat_and_speak(agent, user_input, instruct=None, sample_rate=22050):
    """RAG 生成回复 + 情感语气 + TTS 流式播放"""
    # 1. RAG 生成文本 + 自动情感语气
    response_text, auto_instruct = agent.chat_with_emotion(user_input)
    # 手动指定的 instruct 优先，否则用情感自动生成的
    final_instruct = instruct or auto_instruct
    print(f"天依 [{final_instruct}]: {response_text}")

    if not response_text.strip():
        return

    # 2. TTS 流式合成 + 播放
    try:
        resp = tts_stream(response_text, instruct=instruct)
        play_stream(resp, sample_rate)
    except requests.ConnectionError:
        print("  [TTS 服务未连接，请确认 CosyVoice server 已启动]")
    except Exception as e:
        print(f"  [TTS 错误: {e}]")


def main():
    parser = argparse.ArgumentParser(description="洛天依语音对话客户端")
    parser.add_argument(
        "--instruct",
        "-i",
        type=str,
        default=None,
        help='全局语气指令，如 "用开心的语气说"',
    )
    parser.add_argument(
        "--tts-server",
        type=str,
        default=None,
        help="TTS 服务地址，默认 http://localhost:9880",
    )
    parser.add_argument(
        "--no-voice", action="store_true", help="仅文本模式，不调用 TTS"
    )
    args = parser.parse_args()

    global TTS_SERVER
    if args.tts_server:
        TTS_SERVER = args.tts_server

    print("=" * 60)
    print("洛天依语音对话 (RAG + CosyVoice TTS)")
    print("=" * 60)

    # 初始化 RAG Agent
    print("正在连接天依核心系统...")
    agent = CompanionAgent(use_emotional_mode=True)

    # 获取采样率
    sample_rate = 22050
    if not args.no_voice:
        try:
            sample_rate = get_sample_rate()
            print(f"TTS 服务已连接 (采样率: {sample_rate})")
        except Exception:
            print("TTS 服务未连接，将仅输出文本")
            args.no_voice = True

    if not HAS_PYAUDIO and not args.no_voice:
        print("提示: 未安装 pyaudio，音频将保存为文件而非实时播放")
        print("  安装: pip install pyaudio")

    print("\n天依上线啦！")
    print("命令: 'exit' 退出 | 'instruct:xxx' 临时设置语气\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("天依: 下次见哟！")
                break

            # 支持临时语气指令: "instruct:用开心的语气说 你好啊"
            instruct = args.instruct
            if user_input.startswith("instruct:"):
                parts = user_input[len("instruct:") :].strip()
                sep = parts.find(" ")
                if sep > 0:
                    instruct = parts[:sep]
                    user_input = parts[sep + 1 :]
                else:
                    print("格式: instruct:语气指令 对话内容")
                    continue

            if args.no_voice:
                response_text = agent.chat(user_input)
                print(f"天依: {response_text}")
            else:
                chat_and_speak(
                    agent, user_input, instruct=instruct, sample_rate=sample_rate
                )

        except KeyboardInterrupt:
            print("\n天依: 下次见哟！")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
