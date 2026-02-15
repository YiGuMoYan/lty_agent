import json
import os
import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from rag_core.utils.logger import logger
from rag_core.llm.llm_client import LLMClient
from rag_core.emotions.emotional_memory import EmotionalMemory
from rag_core.generation.live2d_generator import Live2DParamGenerator
from rag_core.generation.unified_generator import UnifiedResponseGenerator
from rag_core.generation.response_style import StyleManager, ResponseStyle, parse_style_from_string
from rag_core.agent.rag_orchestrator import RagOrchestrator
from emotion_live2d_map import Live2DSmoother
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
    MAX_TOKENS = 4000  # 最大 token 数限制

    def __init__(self, use_emotional_mode=True, style: Optional[str] = None, use_unified_generator=True):
        self.client = LLMClient.get_instance()
        self.live2d_generator = Live2DParamGenerator()
        self.use_emotional_mode = use_emotional_mode
        self.use_unified_generator = use_unified_generator  # 是否使用统一生成器

        # 初始化 RAG 编排器 (负责情感分析、路由和工具执行)
        self.orchestrator = RagOrchestrator(use_emotional_mode=use_emotional_mode)

        # 初始化 Live2D 平滑器 (默认 alpha 会在调用时通过 dynamic_alpha 覆盖)
        self.smoother = Live2DSmoother(alpha=0.6)

        if style:
            try:
                initial_style = parse_style_from_string(style)
            except ValueError:
                logger.warning(f"无效的风格: {style}, 使用默认风格")
                initial_style = ResponseStyle.CASUAL
        else:
            try:
                initial_style = parse_style_from_string(DEFAULT_RESPONSE_STYLE)
            except ValueError:
                initial_style = ResponseStyle.CASUAL

        self.style_manager = StyleManager(default_style=initial_style)

        if use_emotional_mode:
            self.emotional_memory = EmotionalMemory()
        else:
            self.emotional_memory = None

        self.history = []

        # System Prompt 缓存
        self._cached_base_prompt: Optional[str] = None  # 缓存基础 prompt（不含情感上下文）
        self._last_emotion_state: Optional[str] = None  # 上次的情感状态标识
        self._last_built_prompt: Optional[str] = None  # 上次构建的完整 prompt

        # Load base system prompt
        prompt_file = PROMPT_PATH
        if use_emotional_mode:
            # rag_core/agent/companion_agent.py -> rag_core/agent -> rag_core -> rag_lty (Root)
            # PROMPT_PATH is usually absolute or relative to root.
            # config.py defines PROMPT_PATH = os.path.join(BASE_DIR, "dataset", "prompts", "system_prompt.txt")
            emotional_prompt_path = os.path.join(os.path.dirname(PROMPT_PATH), "SYSTEM_PROMPT_EMOTIONAL")
            if os.path.exists(emotional_prompt_path):
                prompt_file = emotional_prompt_path
                logger.info("使用情感陪伴模式")

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.base_system_prompt = f.read()
        except Exception as e:
            logger.warning(f"无法加载提示词文件 {prompt_file}: {e}")
            self.base_system_prompt = "你是洛天依。"

        # Build initial system prompt and add to history
        self.history.append({"role": "system", "content": self._build_system_prompt(None)})

        # 初始化统一生成器（如果启用）
        if self.use_unified_generator:
            self.unified_generator = UnifiedResponseGenerator(self.base_system_prompt)
            logger.info("启用统一生成模式（对话+Live2D一次生成）")

    async def initialize(self):
        """Asynchronous initialization"""
        if self.use_emotional_mode and self.emotional_memory:
            await self.emotional_memory.initialize()
            logger.info("情感记忆系统初始化完成")

    def _build_system_prompt(self, emotion_state) -> str:
        """动态构建system prompt，融合基础prompt + 关系状态 + 情感上下文

        优化：缓存不含情感上下文的基础部分，只有情感状态变化时才重建完整 prompt
        """
        # 生成当前情感状态标识
        current_emotion_key = None
        if emotion_state:
            current_emotion_key = f"{emotion_state.primary_emotion}:{emotion_state.intensity:.2f}"

        # 如果情感状态没变化，直接返回缓存的完整 prompt
        if current_emotion_key == self._last_emotion_state and self._last_built_prompt:
            return self._last_built_prompt

        # 情感状态变化（或首次构建），需要重建
        # 1. 获取或构建缓存的基础 prompt（不含情感上下文）
        if self._cached_base_prompt is None:
            self._cached_base_prompt = self._build_base_prompt()

        # 2. 构建情感上下文
        emotion_context = ""
        if emotion_state:
            emotion_context = f"【当前情感上下文】\n"
            emotion_context += f"情感: {emotion_state.primary_emotion}(强度:{emotion_state.intensity:.2f})\n"
            if emotion_state.triggers and emotion_state.triggers != ["日常"]:
                emotion_context += f"触发因素: {', '.join(emotion_state.triggers)}\n"
            emotion_context += "\n"

        # 3. 拼接完整 prompt
        full_prompt = self._cached_base_prompt + emotion_context

        # 4. 更新缓存
        self._last_emotion_state = current_emotion_key
        self._last_built_prompt = full_prompt

        return full_prompt

    def _build_base_prompt(self) -> str:
        """构建基础 prompt（不含情感上下文）"""
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

        parts.append(self.base_system_prompt)
        return "".join(parts)

    async def _summarize_history(self):
        """滚动总结历史对话 (Async)"""
        if len(self.history) <= 25:
            return

        # Check if we can summarize (needs emotional memory)
        if not (self.use_emotional_mode and self.emotional_memory):
            return

        logger.info("触发滚动总结...")
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

                await self.emotional_memory._save_profile()
                logger.info(f"已生成滚动总结: {summary_text[:50]}...")

                # Remove these 10 turns from history
                # Keep system prompt (0) and append the rest (11:)
                self.history = [self.history[0]] + self.history[11:]

        except Exception as e:
            logger.error(f"滚动总结失败: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """简单估算 token 数量（中英文混合估算）"""
        if not text:
            return 0
        # 简单估算：中文约每字2token，英文约每4字符1token
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english = len(text) - chinese
        return chinese * 2 + english // 4

    async def _trim_history(self):
        """管理上下文窗口，避免历史记录无限增长"""
        # Try summarizing first
        await self._summarize_history()

        if len(self.history) > 1:
            # 始终保留 system prompt (index 0)
            system_prompt = self.history[0]
            messages = self.history[1:]

            # 1. 先按轮数限制
            if len(messages) > self.MAX_HISTORY_TURNS:
                messages = messages[-self.MAX_HISTORY_TURNS:]

            # 2. 再按 token 数限制
            total_tokens = self._estimate_tokens(system_prompt.get("content", ""))
            trimmed_messages = []
            for msg in messages:
                msg_tokens = self._estimate_tokens(msg.get("content", ""))
                if total_tokens + msg_tokens > self.MAX_TOKENS:
                    break
                total_tokens += msg_tokens
                trimmed_messages.append(msg)

            self.history = [system_prompt] + trimmed_messages

    async def chat(self, user_input):
        """
        Process user input: Emotional Analysis -> Intent Routing -> Tool -> Response (Async)
        """
        # 1. Execute RAG Pipeline via Orchestrator
        tool_context, emotion_state, is_pure_emotional = await self.orchestrator.execute(user_input, self.history)

        # 2. 动态构建system prompt（含情感上下文）
        if self.use_emotional_mode and emotion_state:
            self.history[0] = {"role": "system", "content": self._build_system_prompt(emotion_state)}

        # 3. 构建user message: 原始输入 + 工具结果
        full_user_msg = user_input
        if tool_context:
            full_user_msg += tool_context

        time_str = datetime.now().strftime("[%H:%M]")
        self.history.append({"role": "user", "content": f"{time_str} {full_user_msg}"})

        logger.info("Generating response...")
        import time as _time
        _llm_start = _time.perf_counter()
        response_msg = await self.client.chat_with_tools(self.history)
        _llm_elapsed = _time.perf_counter() - _llm_start
        logger.debug(f"LLM 生成耗时: {_llm_elapsed:.3f}s")

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
                # DB ops are async now
                await self.emotional_memory.store_emotional_context(
                    emotion_state=emotion_state,
                    user_input=user_input,
                    ai_response=final_answer,
                )
                logger.info("已保存情感记忆")
            except Exception as e:
                logger.error(f"保存情感记忆失败: {e}")

        self.history.append({"role": "assistant", "content": f"{time_str} {final_answer}"})

        # 5. 上下文截断
        await self._trim_history()

        return final_answer

    async def chat_with_live2d_unified(self, user_input):
        """
        统一生成模式：一次LLM调用同时生成对话和Live2D参数 (Async)
        返回: (回复文本, 语气指令, EmotionState, live2d_data)
        """
        # 1. 输入验证
        validated_input = self._validate_input(user_input)
        if not validated_input:
            return None

        # 2. 执行 RAG Pipeline
        tool_context, emotion_state = await self._execute_rag_pipeline(validated_input)

        # 3. 更新 system prompt
        if self.use_emotional_mode and emotion_state:
            self._update_system_prompt(emotion_state)

        # 4. 构建消息并生成回复
        full_user_msg = self._build_user_message(validated_input, tool_context)
        unified_result = await self._generate_unified_response(full_user_msg, emotion_state)

        # 5. 处理结果并返回
        return await self._process_unified_result(unified_result, emotion_state, full_user_msg)

    def _validate_input(self, user_input: str) -> Optional[str]:
        """验证并清理用户输入"""
        if not user_input or len(user_input.strip()) == 0:
            return None

        # 限制输入长度
        max_length = 2000
        if len(user_input) > max_length:
            user_input = user_input[:max_length]

        # 过滤危险字符（空字符等）
        return user_input.replace("\x00", "").strip()

    async def _execute_rag_pipeline(self, user_input: str) -> Tuple[Optional[str], "EmotionState"]:
        """执行 RAG 编排管道"""
        from rag_core.routers.emotional_router import EmotionState

        tool_context, emotion_state, is_pure_emotional = await self.orchestrator.execute(user_input, self.history)

        # Ensure emotion_state exists for fallback/generation
        if not emotion_state:
            emotion_state = EmotionState(
                primary_emotion="平静", intensity=0.3, confidence=0.5,
                context=user_input, triggers=["日常"], timestamp=""
            )

        return tool_context, emotion_state

    def _update_system_prompt(self, emotion_state: "EmotionState") -> None:
        """更新 system prompt"""
        from rag_core.routers.emotional_router import EmotionState
        self.history[0] = {"role": "system", "content": self._build_system_prompt(emotion_state)}
        # 同步更新unified_generator的system prompt
        if hasattr(self, 'unified_generator'):
            from rag_core.generation.unified_generator import LIVE2D_INSTRUCTION
            self.unified_generator.enhanced_system_prompt = self._build_system_prompt(emotion_state) + LIVE2D_INSTRUCTION

    def _build_user_message(self, user_input: str, tool_context: Optional[str]) -> str:
        """构建用户消息"""
        full_user_msg = user_input
        if tool_context:
            full_user_msg += tool_context
        return full_user_msg

    async def _generate_unified_response(
        self,
        full_user_msg: str,
        emotion_state: "EmotionState"
    ) -> Dict[str, Any]:
        """调用统一生成器生成回复"""
        from rag_core.routers.emotional_router import EmotionState
        import time as _time

        _start = _time.perf_counter()

        # 构建messages（不含system，由unified_generator添加）
        time_str = datetime.now().strftime("[%H:%M]")
        messages = self.history[1:] + [{"role": "user", "content": f"{time_str} {full_user_msg}"}]

        unified_result = await self.unified_generator.generate(
            messages=messages,
            emotion=emotion_state.primary_emotion,
            intensity=emotion_state.intensity
        )

        _elapsed = _time.perf_counter() - _start
        logger.debug(f"统一生成总耗时: {_elapsed:.3f}s")

        return unified_result

    async def _generate_response(
        self,
        full_user_msg: str,
        emotion_state: "EmotionState"
    ) -> str:
        """
        只负责LLM生成，不执行RAG。
        用于统一生成失败后的回退，避免重复RAG流程。
        """
        from rag_core.routers.emotional_router import EmotionState

        time_str = datetime.now().strftime("[%H:%M]")
        self.history.append({"role": "user", "content": f"{time_str} {full_user_msg}"})

        # 直接调用LLM生成
        response_msg = await self.client.chat_with_tools(self.history)

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

        # 存储助手回复到历史
        self.history.append({"role": "assistant", "content": f"{time_str} {base_answer}"})

        # 存储情感记忆
        if self.use_emotional_mode and emotion_state and self.emotional_memory:
            try:
                await self.emotional_memory.store_emotional_context(
                    emotion_state=emotion_state,
                    user_input=full_user_msg,
                    ai_response=base_answer,
                )
                logger.info("已保存情感记忆")
            except Exception as e:
                logger.error(f"保存情感记忆失败: {e}")

        return base_answer

    async def _process_unified_result(
        self,
        unified_result: Dict[str, Any],
        emotion_state: "EmotionState",
        full_user_msg: str
    ) -> Tuple[str, str, "EmotionState", Dict[str, Any]]:
        """处理统一生成结果"""
        from rag_core.routers.emotional_router import EmotionState
        time_str = datetime.now().strftime("[%H:%M]")
        emotion = emotion_state.primary_emotion

        if unified_result:
            text = unified_result["text"]
            live2d_data = unified_result["live2d"]

            # Apply Smoothing to params (动态调整 alpha：根据情感强度)
            if "params" in live2d_data:
                intensity = emotion_state.intensity if emotion_state else 0.3
                # 提高 alpha 范围 (0.6-0.9)：前端已有600ms easeOut过渡，后端需要更快响应
                # 强度越高，alpha越大（响应更快）；强度越低，alpha越小
                dynamic_alpha = max(0.6, 0.9 - intensity * 0.3)
                live2d_data["params"] = self.smoother.smooth(live2d_data["params"], alpha=dynamic_alpha)

            # 更新历史
            self.history.append({"role": "user", "content": f"{time_str} {full_user_msg}"})
            self.history.append({"role": "assistant", "content": f"{time_str} {text}"})

            # 保存情感记忆
            if self.use_emotional_mode and emotion_state and self.emotional_memory:
                try:
                    await self.emotional_memory.store_emotional_context(
                        emotion_state=emotion_state,
                        user_input=full_user_msg,
                        ai_response=text
                    )
                except Exception as e:
                    logger.error(f"保存情感记忆失败: {e}")

        else:
            # Fallback: 分离生成（不重复RAG，直接调用LLM）
            logger.warning("统一生成失败，回退到分离模式")
            # 直接调用 _generate_response，避免重复执行 RAG 流程
            text = await self._generate_response(full_user_msg, emotion_state)
            live2d_data = self.generate_live2d_params(text, emotion, emotion_state.intensity)

            # Apply Smoothing (generate_live2d_params might return unsmoothed data)
            # 动态调整 alpha：根据情感强度
            # 提高 alpha 范围 (0.6-0.9)：前端已有600ms easeOut过渡，后端需要更快响应
            if "params" in live2d_data:
                intensity = emotion_state.intensity if emotion_state else 0.3
                dynamic_alpha = max(0.6, 0.9 - intensity * 0.3)
                live2d_data["params"] = self.smoother.smooth(live2d_data["params"], alpha=dynamic_alpha)

            # 历史和记忆已在 _generate_response 中处理

        # Trim History
        await self._trim_history()

        instruct = self.EMOTION_INSTRUCT_MAP.get(emotion, self.EMOTION_INSTRUCT_MAP["平静"])
        return text, instruct, emotion_state, live2d_data

    def set_style(self, style: str) -> bool:
        try:
            parsed_style = parse_style_from_string(style)
            self.style_manager.set_style(parsed_style)
            logger.info(f"风格已切换为: {parsed_style.value}")
            return True
        except ValueError as e:
            logger.error(f"风格切换失败: {e}")
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

        logger.warning("LLM Live2D 生成失败，回退到静态映射")
        from emotion_live2d_map import get_live2d_params
        fallback = get_live2d_params(emotion, intensity)
        logger.debug(f"静态映射结果: emotion={emotion}, params数={len(fallback['params'])}")
        return fallback
