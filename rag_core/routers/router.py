import json
import time
from typing import Optional, Dict, Any, List
from rag_core.llm.llm_client import LLMClient
from rag_core.knowledge.rag_tools import TOOLS_SCHEMA
from rag_core.utils.logger import logger

# 意图缓存配置
INTENT_CACHE_TTL = 300  # 5分钟
INTENT_CACHE_MAX_SIZE = 100

class IntentCache:
    """意图路由缓存"""
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}

    def _normalize_query(self, query: str) -> str:
        """标准化查询，用于缓存匹配"""
        # 去除空格、标点，转小写
        import re
        normalized = re.sub(r'[^\w\u4e00-\u9fff]', '', query)
        return normalized.lower()

    def _cleanup_expired(self):
        """主动清理所有过期条目"""
        current_time = time.time()
        expired_keys = [
            key for key, ts in self._timestamps.items()
            if current_time - ts >= INTENT_CACHE_TTL
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
        if expired_keys:
            logger.debug(f"[IntentCache] Cleaned up {len(expired_keys)} expired entries")

    def get(self, query: str) -> Optional[Dict]:
        """获取缓存的意图结果"""
        # 随机清理：10%概率触发主动清理，避免字典持续增长
        if len(self._cache) > INTENT_CACHE_MAX_SIZE // 2 and hash(query) % 10 == 0:
            self._cleanup_expired()

        key = self._normalize_query(query)
        if key in self._cache:
            # 检查是否过期
            if time.time() - self._timestamps[key] < INTENT_CACHE_TTL:
                return self._cache[key]
            else:
                # 过期清理
                self._cache.pop(key, None)
                self._timestamps.pop(key, None)
        return None

    def set(self, query: str, result: Dict):
        """缓存意图结果"""
        # 缓存满了时，清理所有过期项后再添加
        if len(self._cache) >= INTENT_CACHE_MAX_SIZE:
            self._cleanup_expired()

        # 如果清理后仍然满了，执行LRU淘汰
        if len(self._cache) >= INTENT_CACHE_MAX_SIZE:
            oldest_key = min(self._timestamps, key=self._timestamps.get)
            self._cache.pop(oldest_key, None)
            self._timestamps.pop(oldest_key, None)

        key = self._normalize_query(query)
        self._cache[key] = result
        self._timestamps[key] = time.time()

# 全局缓存实例
_intent_cache = IntentCache()

class IntentRouter:
    def __init__(self):
        self.client = LLMClient.get_instance()

    async def route(self, user_query: str, history: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """
        Determine if the query needs tools.
        优化：添加意图缓存机制

        Args:
            user_query: 用户输入的查询
            history: 对话历史（可选）

        Returns:
            Optional[Dict[str, Any]]: { "tool": "name", "args": {...} } or None
        """
        # 1. 检查缓存
        cached_result = _intent_cache.get(user_query)
        if cached_result is not None:
            logger.debug(f"[Router] 缓存命中: {user_query[:20]}... -> {cached_result.get('tool')}")
            return cached_result

        # 2. 正常路由逻辑
        result = await self._do_route(user_query, history)

        # 3. 缓存结果（仅缓存有效的工具调用结果）
        if result and result.get("tool"):
            _intent_cache.set(user_query, result)

        return result

    async def _do_route(self, user_query: str, history: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """执行实际的路由逻辑

        Args:
            user_query: 用户输入的查询
            history: 对话历史（可选）

        Returns:
            Optional[Dict[str, Any]]: 路由结果，包含 tool 和 args
        """
        import datetime
        current_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.datetime.now().year

        # Build Context String from last 5 turns
        context_str = "None"
        if history:
            # Get last user and assistant message (Commercial Standard: 5 turns)
            # Python slice is safe even if len < 5
            last_turns = history[-5:]
            context_str = json.dumps(last_turns, ensure_ascii=False)
            # Strict prompting for 7B models
        system_prompt = (
            f"You are the Intent Router for Luo Tianyi. Current Date: {current_date_str}. Current Year: {current_year}.\n"
            "STRICT RULES:\n"
            "1. ALWAYS output JSON only. Format: {\"tool\": \"name\", \"args\": {...}}\n"
            "2. DO NOT use paths. Use EXACT tool names.\n"
            "3. DO NOT TRANSLATE Chinese entity names.\n"
            "4. 'who wrote'/'lyrics' -> 'search_lyrics'.\n"
            "5. 'Tell me more'/'context'/'meaning' -> 'search_knowledge_base'.\n"
            "6. 'concert'/'tour'/'event' OR queries with TIME (last year, 2025) -> 'query_knowledge_graph'.\n"
            "7. Resolve relative time: 'last year' -> {current_year - 1}; 'this year' -> {current_year}.\n"
            "8. If no tool needed -> {\"tool\": null}.\n\n"
            f"Recent History: {context_str}\n\n"
            "EXAMPLES:\n"
            "   # 1. Lyrics & Music Info\n"
            "   User: '你知道为了你唱下去这首歌吗' -> {\"tool\": \"search_lyrics\", \"args\": {\"song_title\": \"为了你唱下去\"}}\n"
            "   User: '勾指起誓谁写的' -> {\"tool\": \"search_lyrics\", \"args\": {\"song_title\": \"勾指起誓\"}}\n"
            "   User: '歌词里有那句 机械的心率' -> {\"tool\": \"search_lyrics\", \"args\": {\"lyrics_snippet\": \"机械的心率\"}}\n"
            "   User: '达拉崩吧的歌词' -> {\"tool\": \"search_lyrics\", \"args\": {\"song_title\": \"达拉崩吧\"}}\n"
            "   User: 'ilem有什么作品' -> {\"tool\": \"search_lyrics\", \"args\": {\"artist_name\": \"ilem\"}}\n\n"

            "   # 2. Entity Facts & Biography\n"
            "   User: 'ilem是谁' -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"ilem\", \"relation_type\": \"artist\"}}\n"
            "   User: '介绍一下洛天依' -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"洛天依\", \"relation_type\": \"introduction\"}}\n"
            "   User: '禾念的CEO是谁' -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"禾念\", \"relation_type\": \"ceo\"}}\n\n"

            "   # 3. Contextual Follow-up (Deep Dive)\n"
            "   User: '讲讲' (Context: ilem) -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"ilem background details\"}}\n"
            "   User: '细说一下' (Context: 演唱会) -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"concert detailed description\"}}\n"
            "   User: '是对你写的吗' (Context: 某首歌) -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"meaning and perspective of the song\"}}\n\n"

            "   # 4. Events & Time (Crucial)\n"
            f"   User: '去年开了什么演唱会' -> {{\"tool\": \"query_knowledge_graph\", \"args\": {{\"entity_name\": \"{current_year - 1}\", \"relation_type\": \"happened_in\"}}}}\n"
            f"   User: '今年的活动' -> {{\"tool\": \"query_knowledge_graph\", \"args\": {{\"entity_name\": \"{current_year}\", \"relation_type\": \"event\"}}}}\n"
            "   User: '在哪里举办的' (Context: concert) -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"concert location detail\"}}\n\n"

            "   # 5. Chitchat & Negation\n"
            "   User: '你好呀' -> {\"tool\": null}\n"
            "   User: '今天天气不错' -> {\"tool\": null}\n"
            "   User: '你不理我了吗' -> {\"tool\": null}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        logger.info(f"[Router] Analyzing: {user_query}")
        try:
            # Force JSON mode if supported, or just rely on prompt
            response = await self.client.client.chat.completions.create(
                model=self.client.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1 # Deterministic
            )
            content = response.choices[0].message.content
            logger.debug(f"[Router] Raw Logic: {content}")

            result = json.loads(content)
            if result.get("tool"):
                return result
            return None

        except Exception as e:
            logger.error(f"[Router] Error: {e}")
            return None
