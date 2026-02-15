import asyncio
import json
import re
from typing import Optional, Dict, Tuple, Any, List, Set

from rag_core.routers.router import IntentRouter
from rag_core.routers.emotional_router import EmotionalRouter, EmotionState
from rag_core.knowledge.rag_tools import AVAILABLE_TOOLS, search_knowledge_base

class RagOrchestrator:
    """
    RAG Orchestrator
    负责协调情感分析、意图路由和工具执行（含 DeepSearch）
    """
    def __init__(self, use_emotional_mode=True):
        self.router = IntentRouter()
        self.use_emotional_mode = use_emotional_mode
        if use_emotional_mode:
            self.emotional_router = EmotionalRouter()
        else:
            self.emotional_router = None

    async def execute(self, user_input: str, history: list) -> Tuple[str, Optional[EmotionState], bool]:
        """
        执行 RAG 流程：并行情感分析与意图路由 -> 工具执行 -> 返回上下文
        Returns: (tool_context, emotion_state, is_pure_emotional)
        """
        # 1. 并行执行情感分析和意图路由
        emotion_task = self._analyze_emotion_safe(user_input, history)
        route_task = self._route_intent_safe(user_input, history)

        emotion_state, route_result = await asyncio.gather(emotion_task, route_task)

        # 2. 判断是否为纯情感倾诉
        is_pure_emotional = False
        if self.use_emotional_mode and self.emotional_router and emotion_state:
            is_pure_emotional = self.emotional_router.is_pure_emotional_query(user_input, emotion_state)
            if is_pure_emotional:
                print(f"[RAG] 纯情感倾诉: {emotion_state.primary_emotion}(强度:{emotion_state.intensity:.2f})")

        # 3. 执行工具（如果不是纯情感倾诉且路由到了工具）
        tool_context = ""
        if not is_pure_emotional and route_result and route_result.get("tool"):
            tool_context = await self._execute_tool_and_deepsearch(route_result)

        return tool_context, emotion_state, is_pure_emotional

    async def _analyze_emotion_safe(self, user_input: str, history: list) -> Optional[EmotionState]:
        if not (self.use_emotional_mode and self.emotional_router):
            return None
        try:
            return await self.emotional_router.analyze_emotion(user_input, history)
        except Exception as e:
            print(f"[RAG] 情感分析失败: {e}")
            return None

    async def _route_intent_safe(self, user_input: str, history: list) -> Optional[Dict]:
        try:
            return await self.router.route(user_input, history=history)
        except Exception as e:
            print(f"[RAG] 意图路由失败: {e}")
            return None

    async def _execute_tool_and_deepsearch(self, route_result: Dict) -> str:
        func_name = route_result["tool"]
        func_args = route_result.get("args", {})
        print(f"[RAG] Intent detected: {func_name}({func_args})")

        if func_name not in AVAILABLE_TOOLS:
            return ""

        function_to_call = AVAILABLE_TOOLS[func_name]
        try:
            # 运行工具（在线程池中，防止阻塞）
            loop = asyncio.get_running_loop()
            tool_result = await loop.run_in_executor(None, lambda: function_to_call(**func_args))

            # --- DeepSearch (Multi-hop) ---
            tool_result = await self._perform_deep_search(tool_result, func_args, func_name)

            return tool_result

        except Exception as e:
            return f"\n\n【共鸣雷达报错】{str(e)}"

    async def _perform_deep_search(self, tool_result: Any, func_args: Dict, func_name: str) -> str:
        """
        处理工具返回结果，执行 DeepSearch（递归查找）并格式化输出
        """
        loop = asyncio.get_running_loop()

        # 解析候选实体
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
                elif isinstance(item, dict):
                    if "result" in item: candidates.add(item["result"])
                    elif "song_title" in item: candidates.add(item["song_title"])

        # 正则提取书名号内容
        text_content = str(tool_result)
        matches = re.findall(r'[「《](.*?)[」》]', text_content)
        for m in matches:
            candidates.add(m)

        # 移除原始查询词
        original_query = func_args.get('entity_name', '') or func_args.get('query', '')
        if original_query in candidates:
            candidates.remove(original_query)

        final_candidates = list(candidates)[:2]

        # 二次查询
        if final_candidates:
            print(f"[RAG] DeepSearch detected entities: {final_candidates}. Triggering recursive lookup...")

            async def fetch_extra(entity):
                if len(entity) < 2: return ""
                res = await loop.run_in_executor(None, lambda: search_knowledge_base(query=entity))
                if res and len(res) > 50 and "[]" not in res:
                    return f"\\n\\n【关联档案：{entity}】\\n{res}"
                return ""

            extra_results = await asyncio.gather(*[fetch_extra(e) for e in final_candidates])
            tool_result = str(tool_result) + "".join(extra_results)

        # 结果验证与格式化
        is_empty = False
        if not parsed_res and not isinstance(parsed_res, list):
            if isinstance(parsed_res, dict) and parsed_res.get("status") == "not_found":
                is_empty = True
            elif len(parsed_res) == 0 and not isinstance(parsed_res, (str, int, float, bool)): # check if truly empty structure
                 # careful here: [] is len 0, {} is len 0
                 # if parsed_res is [], len is 0.
                 is_empty = True

        # Special handling for empty string result
        if not tool_result:
            is_empty = True

        tool_context = ""
        if is_empty:
            if func_name == "query_knowledge_graph":
                print(f"[RAG] Graph failed. Last resort: KB Search.")
                kb_res = await loop.run_in_executor(None, lambda: search_knowledge_base(query=func_args.get('entity_name', '')))
                if kb_res and kb_res != "[]":
                    tool_context = f"\n\n【共鸣雷达补救】\n原图谱查询失败，但在档案库中发现：\n{kb_res}\n(请回答)"
                else:
                    tool_context = f"\n\n【共鸣雷达反馈】\n结果: 未找到任何相关数据。\n[系统指令] 严禁编造。"
            else:
                tool_context = f"\n\n【共鸣雷达反馈】\n结果: 未找到任何相关数据。\n[系统指令] 严禁编造。"
        else:
            tool_context = f"\n\n【共鸣雷达数据】\n工具调用: {func_name}\n检索结果: {tool_result}\n(请根据以上真实数据回答用户。)"

        return tool_context
