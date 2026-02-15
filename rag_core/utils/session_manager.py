"""
会话管理器 - Session Manager
管理 WebSocket 连接中的用户会话
"""

import uuid
from typing import Dict, Optional
from rag_core.agent.companion_agent import CompanionAgent
from rag_core.utils.logger import logger


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, CompanionAgent] = {}

    def create_session(self) -> str:
        """创建新会话并返回 session_id"""
        session_id = str(uuid.uuid4())
        # 为每个会话创建独立的 Agent 实例
        agent = CompanionAgent(use_emotional_mode=True, use_unified_generator=True)
        self.sessions[session_id] = agent
        logger.info(f"[SessionManager] Created session: {session_id}")
        return session_id

    def get_agent(self, session_id: str) -> Optional[CompanionAgent]:
        """获取会话对应的 Agent"""
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        """移除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"[SessionManager] Removed session: {session_id}")


# 全局实例
session_manager = SessionManager()
