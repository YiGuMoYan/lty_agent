import json
import os
import asyncio
from typing import Optional, Dict
from datetime import datetime
from rag_core.llm.llm_client import LLMClient
from rag_core.knowledge.rag_tools import AVAILABLE_TOOLS
from rag_core.routers.router import IntentRouter
from rag_core.routers.emotional_router import EmotionalRouter
from rag_core.emotions.emotional_memory import EmotionalMemory
from rag_core.generation.live2d_generator import Live2DParamGenerator
from rag_core.generation.unified_generator import UnifiedResponseGenerator
from rag_core.generation.response_style import StyleManager, ResponseStyle, parse_style_from_string
from config import PROMPT_PATH, DEFAULT_RESPONSE_STYLE

class CompanionAgent:
    # 用户情感 → 回复语气指令映射（天依应该用什么语气回应）
    EMOTION_INSTRUCT_MAP = {
        "开心": "用开心的语气说这句话",
        "难过": "用温柔安慰的语气说这句话",
        "焦虑": "用轻柔舒缓的语气说这句话",
        "孤独": "用温暖陪伴的语气说这句话",
        "愤怒": "用冷静温和的语气说这句话",
        "疲惫": "用轻柔关心的语气说这句话",
        "困惑": "用耐心温和的语气说这句话",
        "平静": "用平静温柔的语气说这句话",
    }

    MAX_HISTORY_TURNS = 30 # Increased for rolling summary buffer

    def __init__(self, use_emotional_mode=True, style: Optional[str] = None, use_unified_generator=True):
        self.client = LLMClient()
        self.router = IntentRouter()
        self.live2d_generator = Live2DParamGenerator()
        self.use_emotional_mode = use_emotional_mode
        self.use_unified_generator = use_unified_generator  # 是否使用统一生成器

        if style:
            try:
                initial_style = parse_style_from_string(style)
            except ValueError:
                print(f"[CompanionAgent] 无效的风格: {style}, 使用默认风格")
                initial_style = ResponseStyle.CASUAL
        else:
            try:
                initial_style = parse_style_from_string(DEFAULT_RESPONSE_STYLE)
            except ValueError:
                initial_style = ResponseStyle.CASUAL

        self.style_manager = StyleManager(default_style=initial_style)

        if use_emotional_mode:
            self.emotional_router = EmotionalRouter()
            self.emotional_memory = EmotionalMemory()
        else:
            self.emotional_router = None
            self.emotional_memory = None

        self.history = []

        # Load base system prompt
        prompt_file = PROMPT_PATH
        if use_emotional_mode:
            # rag_core/agent/companion_agent.py -> rag_core/agent -> rag_core -> rag_lty (Root)
            # PROMPT_PATH is usually absolute or relative to root.
            # config.py defines PROMPT_PATH = os.path.join(BASE_DIR, "dataset", "prompts", "system_prompt.txt")
            emotional_prompt_path = os.path.join(os.path.dirname(PROMPT_PATH), "SYSTEM_PROMPT_EMOTIONAL")
            if os.path.exists(emotional_prompt_path):
                prompt_file = emotional_prompt_path
                print("[CompanionAgent] 使用情感陪伴模式")

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.base_system_prompt = f.read()
        except Exception as e:
            print(f"[CompanionAgent] Warning: Could not load prompt from {prompt_file}: {e}")
            self.base_system_prompt = "你是洛天依。"

        # Build initial system prompt and add to history
        self.history.append({"role": "system", "content": self._build_system_prompt(None)})

        # 初始化统一生成器（如果启用）
        if self.use_unified_generator:
            self.unified_generator = UnifiedResponseGenerator(self.base_system_prompt)
            print("[CompanionAgent] 启用统一生成模式（对话+Live2D一次生成）")

    def _build_system_prompt(self, emotion_state) -> str:
        """动态构建system prompt，融合基础prompt + 关系状态 + 情感上下文"""
        from datetime import datetime
        current_date = datetime.now().strftime("%Y年%m月%d日")
        time_context = f"【系统时间锚点：当前是 {current_date}】\n(请根据此时间判断'去年'、'今年'等相对时间词)\n\n"

        parts = [time_context]

        # 关系状态
        if self.use_emotional_mode and self.emotional_memory:
            summary = self.emotional_memory.get_profile_summary()
            depth = summary["relationship_depth"]
            parts.append(f"【当前关系状态】\n")
            parts.append(f"互动次数: {summary['total_interactions']}\n")
            parts.append(f"关系深度: {depth:.2f}\n")
            parts.append(f"信任度: {summary['trust_level']:.2f}\n")
            if summary['dominant_emotions']:
                parts.append(f"主要情感: {', '.join([f'{emo}({count})' for emo, count in summary['dominant_emotions']])}\n")

            if depth < 0.3:
                parts.append("关系指导: 保持温柔但适度距离\n")
            elif depth < 0.7:
                parts.append("关系指导: 更自然亲近\n")
            else:
                parts.append("关系指导: 更真诚直接\n")
            parts.append("\n")

            # 滚动总结 (Long-term Memory)
            conv_summary = self.emotional_memory.profile.conversation_summary
            if conv_summary:
                # Limit to last 1000 chars or last 5 entries approx
                display_summary = conv_summary[-1000:]
                if len(conv_summary) > 1000:
                    display_summary = "..." + display_summary
                parts.append(f"【长期记忆 (过往对话摘要)】\n{display_summary}\n\n")

        # 情感上下文
        if emotion_state:
            parts.append(f"【当前情感上下文】\n")
            parts.append(f"情感: {emotion_state.primary_emotion}(强度:{emotion_state.intensity:.2f})\n")
            if emotion_state.triggers and emotion_state.triggers != ["日常"]:
                parts.append(f"触发因素: {', '.join(emotion_state.triggers)}\n")
            parts.append("\n")

        parts.append(self.base_system_prompt)
        return "".join(parts)

    async def _summarize_history(self):
        """滚动总结历史对话 (Async)"""
        if len(self.history) <= 25:
            return

        # Check if we can summarize (needs emotional memory)
        if not (self.use_emotional_mode and self.emotional_memory):
            return

        print("[Companion] 触发滚动总结...")
        # Slice self.history[1:11] (skip system prompt, take 10 turns)
        turns_to_summarize = self.history[1:11]

        # Extract timestamp from first message if possible, else use current
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Prepare text
        conversation_text = ""
        for turn in turns_to_summarize:
            role = "用户" if turn['role'] == "user" else "天依"
            content = turn['content']
            # Remove tool outputs for summary clarity if needed, but keeping simple for now
            conversation_text += f"{role}: {content}\n"

        prompt = f"""请简要总结以下对话片段，作为长期记忆保存。
侧重于用户提到的关键信息、偏好、经历以及天依的情感互动。
不要流水账，要提炼核心内容。
格式要求：[YYYY-MM-DD HH:MM] 总结内容

对话内容：
{conversation_text}"""

        try:
            # Call LLM
            # We use a temporary simple history for this call
            summary_msgs = [{"role": "user", "content": prompt}]
            summary_response = await self.client.chat_with_tools(summary_msgs)

            if summary_response and summary_response.content:
                summary_text = summary_response.content.strip()
                # Ensure timestamp format if LLM missed it
                if not summary_text.startswith("["):
                     summary_text = f"[{timestamp}] {summary_text}"

                # Update profile
                current_summary = self.emotional_memory.profile.conversation_summary
                if current_summary:
                    self.emotional_memory.profile.conversation_summary = current_summary + "\n" + summary_text
                else:
                    self.emotional_memory.profile.conversation_summary = summary_text

                self.emotional_memory._save_profile()
                print(f"[Companion] 已生成滚动总结: {summary_text[:50]}...")

                # Remove these 10 turns from history
                # Keep system prompt (0) and append the rest (11:)
                self.history = [self.history[0]] + self.history[11:]

        except Exception as e:
            print(f"[Companion] 滚动总结失败: {e}")

    async def _trim_history(self):
        """管理上下文窗口，避免历史记录无限增长"""
        # Try summarizing first
        await self._summarize_history()

        if len(self.history) > 1:
            # 始终保留 system prompt (index 0)
            system_prompt = self.history[0]
            remaining = self.history[1:]

            # 保留最近 N 轮
            if len(remaining) > self.MAX_HISTORY_TURNS:
                 self.history = [system_prompt] + remaining[-self.MAX_HISTORY_TURNS:]

    async def _execute_rag_pipeline(self, user_input):
        """
        执行核心 RAG 流程：情感分析 -> 意图路由 -> 工具执行 (含 DeepSearch) -> 返回上下文
        Returns: (tool_context, emotion_state, is_pure_emotional)
        """
        # 0. 情感分析
        emotion_state = None
        is_pure_emotional = False

        if self.use_emotional_mode and self.emotional_router:
            try:
                emotion_state = await self.emotional_router.analyze_emotion(user_input, self.history)
                is_pure_emotional = self.emotional_router.is_pure_emotional_query(user_input, emotion_state)
                if is_pure_emotional:
                    print(f"[Companion] 纯情感倾诉: {emotion_state.primary_emotion}(强度:{emotion_state.intensity:.2f})")
            except Exception as e:
                print(f"[Companion] 情感分析失败: {e}")

        # 1. Intent Routing — 始终执行（除非纯情感倾诉）
        route_result = None
        if not is_pure_emotional:
            route_result = await self.router.route(user_input, history=self.history)

        tool_context = ""

        if route_result and route_result.get("tool"):
            func_name = route_result["tool"]
            func_args = route_result.get("args", {})
            print(f"[Companion] Router detected intent: {func_name}({func_args})")

            if func_name in AVAILABLE_TOOLS:
                function_to_call = AVAILABLE_TOOLS[func_name]
                try:
                    # Note: Tool functions are synchronous for now (unless we refactor rag_tools.py)
                    # Running them in executor to avoid blocking if they are slow
                    loop = asyncio.get_running_loop()
                    tool_result = await loop.run_in_executor(None, lambda: function_to_call(**func_args))

                    # --- DeepSearch (Multi-hop) ---
                    candidates = set()
                    parsed_res = []
                    try:
                        parsed_res = json.loads(tool_result)
                    except:
                        pass

                    if isinstance(parsed_res, list):
                        for item in parsed_res:
                            if isinstance(item, str):
                                candidates.add(item)
                            elif isinstance(item, dict) and "result" in item:
                                candidates.add(item["result"])
                            # Add support for structured objects returned by tools
                            elif isinstance(item, dict) and "song_title" in item:
                                candidates.add(item["song_title"])
                            elif isinstance(item, dict) and "content" in item:
                                # Extract entities from content? Heuristic.
                                pass

                    import re
                    text_content = str(tool_result)
                    matches = re.findall(r'[「《](.*?)[」》]', text_content)
                    for m in matches:
                        candidates.add(m)

                    original_query = func_args.get('entity_name', '') or func_args.get('query', '')
                    if original_query in candidates:
                        candidates.remove(original_query)

                    final_candidates = list(candidates)[:2]

                    if final_candidates:
                        print(f"[Companion] DeepSearch detected entities: {final_candidates}. Triggering recursive lookup...")
                        from rag_core.knowledge.rag_tools import search_knowledge_base
                        extra_info = ""

                        # Parallelize secondary searches
                        async def fetch_extra(entity):
                            if len(entity) < 2: return ""
                            res = await loop.run_in_executor(None, lambda: search_knowledge_base(query=entity))
                            if res and len(res) > 50 and "[]" not in res:
                                return f"\\n\\n【关联档案：{entity}】\\n{res}"
                            return ""

                        extra_results = await asyncio.gather(*[fetch_extra(e) for e in final_candidates])
                        extra_info = "".join(extra_results)

                        if extra_info:
                             tool_result += extra_info
                    # --- End DeepSearch ---

                    # Data Validation
                    is_empty = False
                    if not parsed_res and not isinstance(parsed_res, list):
                         if isinstance(parsed_res, dict) and "status" in parsed_res and parsed_res["status"] == "not_found":
                              is_empty = True
                         elif len(parsed_res) == 0:
                              is_empty = True

                    if is_empty:
                        if func_name == "query_knowledge_graph":
                             print(f"[Companion] Graph failed completely. Last resort: KB Search.")
                             from rag_core.knowledge.rag_tools import search_knowledge_base
                             kb_res = await loop.run_in_executor(None, lambda: search_knowledge_base(query=func_args.get('entity_name', '')))
                             if kb_res and kb_res != "[]":
                                 tool_context = f"\n\n【共鸣雷达补救】\n原图谱查询失败，但在档案库中发现：\n{kb_res}\n(请回答)"
                             else:
                                 tool_context = f"\n\n【共鸣雷达反馈】\n结果: 未找到任何相关数据。\n[系统指令] 严禁编造。"
                        else:
                             tool_context = f"\n\n【共鸣雷达反馈】\n结果: 未找到任何相关数据。\n[系统指令] 严禁编造。"
                    else:
                        tool_context = f"\n\n【共鸣雷达数据】\n工具调用: {func_name}\n检索结果: {tool_result}\n(请根据以上真实数据回答用户。)"

                except Exception as e:
                    tool_context = f"\n\n【共鸣雷达报错】{str(e)}"
            else:
                 print(f"[Companion] Constructing response with tool result...")

        return tool_context, emotion_state, is_pure_emotional

    async def chat(self, user_input):
        """
        Process user input: Emotional Analysis -> Intent Routing -> Tool -> Response (Async)
        """
        # 1. Execute RAG Pipeline
        tool_context, emotion_state, is_pure_emotional = await self._execute_rag_pipeline(user_input)

        # 2. 动态构建system prompt（含情感上下文）
        if self.use_emotional_mode and emotion_state:
            self.history[0] = {"role": "system", "content": self._build_system_prompt(emotion_state)}

        # 3. 构建user message: 原始输入 + 工具结果
        full_user_msg = user_input
        if tool_context:
            full_user_msg += tool_context

        time_str = datetime.now().strftime("[%H:%M]")
        self.history.append({"role": "user", "content": f"{time_str} {full_user_msg}"})

        print(f"[Companion] Generating response...")
        import time as _time
        _llm_start = _time.perf_counter()
        response_msg = await self.client.chat_with_tools(self.history)
        _llm_elapsed = _time.perf_counter() - _llm_start
        print(f"[Companion] LLM 生成耗时: {_llm_elapsed:.3f}s")

        base_answer = ""
        if response_msg:
             answer_content = response_msg.content
             if answer_content is not None:
                 import re
                 base_answer = re.sub(r'[\(（][^\)）]+[\)）]', '', answer_content)
                 base_answer = re.sub(r'\n\s*\n', '\n', base_answer).strip()
             else:
                 base_answer = "（数据流中断...）"
        else:
             base_answer = "（数据流中断...）"

        final_answer = base_answer

        # 4. 存储情感记忆
        if self.use_emotional_mode and emotion_state and self.emotional_memory:
            try:
                # DB ops are sync, run in executor if needed, but sqlite is fast enough for now usually
                # Ideally refactor EmotionalMemory to async
                self.emotional_memory.store_emotional_context(
                    emotion_state=emotion_state,
                    user_input=user_input,
                    ai_response=final_answer,
                )
                print(f"[Companion] 已保存情感记忆")
            except Exception as e:
                print(f"[Companion] 保存情感记忆失败: {e}")

        self.history.append({"role": "assistant", "content": f"{time_str} {final_answer}"})

        # 5. 上下文截断
        await self._trim_history()

        return final_answer

    async def chat_with_live2d_unified(self, user_input):
        """
        统一生成模式：一次LLM调用同时生成对话和Live2D参数 (Async)
        返回: (回复文本, 语气指令, EmotionState, live2d_data)
        """
        from rag_core.routers.emotional_router import EmotionState
        import time as _time

        # 1. Execute RAG Pipeline
        tool_context, emotion_state, is_pure_emotional = await self._execute_rag_pipeline(user_input)

        # Ensure emotion_state exists for fallback/generation
        if not emotion_state:
            emotion_state = EmotionState(
                primary_emotion="平静", intensity=0.3, confidence=0.5,
                context=user_input, triggers=["日常"], timestamp=""
            )
        emotion = emotion_state.primary_emotion

        # 2. 动态更新system prompt
        if self.use_emotional_mode and emotion_state:
            self.history[0] = {"role": "system", "content": self._build_system_prompt(emotion_state)}
            # 同步更新unified_generator的system prompt
            if hasattr(self, 'unified_generator'):
                from rag_core.generation.unified_generator import LIVE2D_INSTRUCTION
                self.unified_generator.enhanced_system_prompt = self._build_system_prompt(emotion_state) + LIVE2D_INSTRUCTION

        # 3. 构建完整user message
        full_user_msg = user_input
        if tool_context:
            full_user_msg += tool_context

        # 4. 使用统一生成器（一次LLM调用）
        _start = _time.perf_counter()

        # 构建messages（不含system，由unified_generator添加）
        time_str = datetime.now().strftime("[%H:%M]")
        messages = self.history[1:] + [{"role": "user", "content": f"{time_str} {full_user_msg}"}]

        unified_result = await self.unified_generator.generate(
            messages=messages,
            emotion=emotion,
            intensity=emotion_state.intensity
        )

        _elapsed = _time.perf_counter() - _start
        print(f"[Companion] 统一生成总耗时: {_elapsed:.3f}s")

        # 5. 处理结果
        if unified_result:
            text = unified_result["text"]
            live2d_data = unified_result["live2d"]

            # 6. 更新历史
            self.history.append({"role": "user", "content": f"{time_str} {full_user_msg}"})
            self.history.append({"role": "assistant", "content": f"{time_str} {text}"})

            # 7. 保存情感记忆
            if self.use_emotional_mode and emotion_state and self.emotional_memory:
                try:
                    self.emotional_memory.store_emotional_context(
                        emotion_state=emotion_state,
                        user_input=user_input,
                        ai_response=text
                    )
                except Exception as e:
                    print(f"[Companion] 保存情感记忆失败: {e}")

        else:
            # Fallback: 分离生成
            print("[Companion] ⚠️ 统一生成失败，回退到分离模式")
            # Note: This will trigger RAG pipeline again in chat() if we called chat() directly
            # But here we already have tool_context, so we should just call LLM directly or implement fallback logic
            # For simplicity, calling self.chat() which re-does RAG is safe but inefficient.
            # Ideally we should split chat() to chat_with_context()
            text = await self.chat(user_input)
            live2d_data = self.generate_live2d_params(text, emotion, emotion_state.intensity)

            # History and memory are handled inside self.chat()

        # 8. Trim History
        await self._trim_history()

        instruct = self.EMOTION_INSTRUCT_MAP.get(emotion, self.EMOTION_INSTRUCT_MAP["平静"])
        return text, instruct, emotion_state, live2d_data

    def set_style(self, style: str) -> bool:
        try:
            parsed_style = parse_style_from_string(style)
            self.style_manager.set_style(parsed_style)
            print(f"[CompanionAgent] 风格已切换为: {parsed_style.value}")
            return True
        except ValueError as e:
            print(f"[CompanionAgent] 风格切换失败: {e}")
            return False

    def get_current_style(self) -> ResponseStyle:
        return self.style_manager.get_current_style()

    def get_available_styles(self) -> Dict[str, str]:
        return self.style_manager.get_available_styles()

    def generate_live2d_params(self, reply_text: str, emotion: str, intensity: float) -> dict:
        """
        生成 Live2D 参数：优先 LLM 生成，失败回退到静态映射。
        (Note: live2d_generator itself is sync currently, could be made async too)
        """
        result = self.live2d_generator.generate(reply_text, emotion, intensity)
        if result is not None:
            return result

        print("[CompanionAgent] LLM Live2D 生成失败，回退到静态映射")
        from emotion_live2d_map import get_live2d_params
        fallback = get_live2d_params(emotion, intensity)
        print(f"[CompanionAgent] 静态映射结果: emotion={emotion}, params数={len(fallback['params'])}")
        return fallback
