"""
会话管理器 - Session Manager
管理 WebSocket 连接中的用户会话
"""

import asyncio
import time
import uuid
from typing import Dict, Optional
from rag_core.agent.companion_agent import CompanionAgent
from rag_core.utils.logger import logger


class SessionManager:
    def __init__(self, session_timeout: int = 3600):
        self.sessions: Dict[str, Dict] = {}  # 存储会话信息字典
        self.session_timeout = session_timeout
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start_cleanup_task(self):
        """启动定期清理任务"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """定期清理过期会话"""
        while True:
            await asyncio.sleep(300)  # 每5分钟检查一次
            await self._cleanup_expired()

    async def _cleanup_expired(self):
        """清理过期会话"""
        current_time = time.time()
        expired = [
            sid for sid, info in self.sessions.items()
            if current_time - info.get("last_active", 0) > self.session_timeout
        ]
        for sid in expired:
            self.remove_session(sid)
        if expired:
            logger.info(f"[SessionManager] Cleaned up {len(expired)} expired sessions")

    def create_session(self, user_id: str = None) -> str:
        """创建新会话并返回 session_id

        Args:
            user_id: 用户ID，如果为None则使用session_id作为user_id
        """
        session_id = str(uuid.uuid4())
        # 使用session_id作为user_id（如果未提供）
        effective_user_id = user_id or session_id
        # 为每个会话创建独立的 Agent 实例
        agent = CompanionAgent(user_id=effective_user_id, use_emotional_mode=True, use_unified_generator=True)
        self.sessions[session_id] = {
            "agent": agent,
            "user_id": effective_user_id,
            "created_at": time.time(),
            "last_active": time.time()
        }
        logger.info(f"[SessionManager] Created session: {session_id}, user_id: {effective_user_id}")
        return session_id

    def get_agent(self, session_id: str) -> Optional[CompanionAgent]:
        """获取会话对应的 Agent"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_active"] = time.time()
            return self.sessions[session_id]["agent"]
        return None

    def remove_session(self, session_id: str):
        """移除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"[SessionManager] Removed session: {session_id}")


# 全局实例
session_manager = SessionManager()


def init_session_manager():
    """初始化并启动会话管理器清理任务"""
    asyncio.create_task(session_manager.start_cleanup_task())
    logger.info("[SessionManager] Cleanup task started")


init_session_manager()
