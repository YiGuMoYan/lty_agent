"""
情感记忆系统 - Emotional Memory
负责存储、管理和检索用户的情感历程，建立长期情感档案
Refactored to use SQLite for scalability.
支持语义检索 (Semantic Retrieval)
"""

import json
import os
import aiosqlite
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from rag_core.routers.emotional_router import EmotionState
from rag_core.utils.logger import logger


# 模块级 embedding 函数缓存
_embedding_function = None


def _get_global_embedding_function():
    """获取全局缓存的 embedding 函数"""
    global _embedding_function
    if _embedding_function is None:
        from rag_core.llm.embeddings import get_embedding_function
        _embedding_function = get_embedding_function()
    return _embedding_function


@dataclass
class EmotionalMemoryEntry:
    """情感记忆条目"""
    timestamp: str
    emotion_state: EmotionState
    user_input: str
    ai_response: str
    interaction_quality: float  # 交互质量评分 0.0-1.0
    similarity: float = 0.0  # 语义相似度 (仅用于检索结果)

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp,
            "emotion_state": asdict(self.emotion_state),
            "user_input": self.user_input,
            "ai_response": self.ai_response,
            "interaction_quality": self.interaction_quality,
            "similarity": self.similarity,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionalMemoryEntry':
        """从字典创建实例（兼容旧数据）"""
        emotion_data = data["emotion_state"]
        # 兼容旧数据：过滤掉 EmotionState 不再接受的字段
        valid_fields = {"primary_emotion", "intensity", "confidence", "context", "triggers", "timestamp"}
        filtered_emotion = {k: v for k, v in emotion_data.items() if k in valid_fields}
        emotion_state = EmotionState(**filtered_emotion)

        return cls(
            timestamp=data["timestamp"],
            emotion_state=emotion_state,
            user_input=data["user_input"],
            ai_response=data["ai_response"],
            interaction_quality=data.get("interaction_quality", 0.5),
        )


@dataclass
class UserEmotionalProfile:
    """用户情感画像"""
    user_id: str
    total_interactions: int
    emotion_distribution: Dict[str, int]  # 情感分布统计
    average_intensity: float  # 平均情感强度
    common_triggers: List[str]  # 常见情感触发因素
    relationship_depth: float  # 关系深度 0.0-1.0
    trust_level: float  # 信任度 0.0-1.0
    last_interaction: str
    emotional_patterns: Dict[str, Any]  # 情感模式
    conversation_summary: str = ""  # 对话滚动总结

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'UserEmotionalProfile':
        # 兼容旧数据：忽略已删除的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class EmotionalMemory:
    def __init__(self, user_id: str = "default_user", memory_dir: str = None):
        self.user_id = user_id

        if memory_dir is None:
            # rag_core/emotions/emotional_memory.py -> rag_core/emotions -> rag_core -> rag_lty (Root)
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            memory_dir = os.path.join(base_dir, "dataset", "emotional_memory")

        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        self.db_path = os.path.join(memory_dir, "emotional_memory.db")

        # 数据库连接（单一连接）
        self._conn = None

        # Initialize with default profile, actual data loaded in initialize()
        self.profile = UserEmotionalProfile(
            user_id=self.user_id,
            total_interactions=0,
            emotion_distribution={},
            average_intensity=0.0,
            common_triggers=[],
            relationship_depth=0.0,
            trust_level=0.0,
            last_interaction="",
            emotional_patterns={},
            conversation_summary=""
        )

        # 批量更新机制
        self._profile_dirty = False  # 标记是否有未保存的更改
        self._update_counter = 0  # 更新计数器
        self._flush_threshold = 5  # 每5次更新写一次DB

        self._initialized = False

    def _get_embedding_fn(self):
        """获取 embedding 函数（使用全局缓存）"""
        return _get_global_embedding_function()

    async def initialize(self):
        """Asynchronous initialization of database and profile"""
        if self._initialized:
            return

        # 创建单一数据库连接
        self._conn = await aiosqlite.connect(self.db_path)

        await self._init_db()
        # Check for legacy file migration
        await self._migrate_from_legacy_files()
        self.profile = await self._load_profile()
        self._initialized = True

    def _get_conn(self):
        """获取数据库连接（返回缓存的连接）"""
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._conn

    async def close(self):
        """关闭数据库连接"""
        # 关闭前确保未保存的profile已写入
        if self._profile_dirty:
            await self._flush_profile()
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("情感记忆数据库连接已关闭")

    async def _init_db(self):
        """Initialize database schema"""
        try:
            conn = await self._get_conn()
            cursor = await conn.cursor()
            # User Profiles Table
            await cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    total_interactions INTEGER DEFAULT 0,
                    emotion_distribution TEXT,
                    average_intensity REAL DEFAULT 0.0,
                    common_triggers TEXT,
                    relationship_depth REAL DEFAULT 0.0,
                    trust_level REAL DEFAULT 0.0,
                    last_interaction TIMESTAMP,
                    emotional_patterns TEXT
                )
            ''')
            # Emotional Memories Table
            await cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotional_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_input TEXT,
                    ai_response TEXT,
                    emotion_state TEXT,
                    interaction_quality REAL,
                    embedding BLOB,
                    FOREIGN KEY(user_id) REFERENCES user_profiles(user_id)
                )
            ''')
            # Index for faster history retrieval
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_user_timestamp ON emotional_memories(user_id, timestamp)')

            # Migration: Check for embedding column
            try:
                await cursor.execute("SELECT embedding FROM emotional_memories LIMIT 1")
            except Exception:
                logger.info("Adding embedding column to emotional_memories")
                await cursor.execute("ALTER TABLE emotional_memories ADD COLUMN embedding BLOB")

            # Migration: Check for conversation_summary column
            try:
                await cursor.execute("SELECT conversation_summary FROM user_profiles LIMIT 1")
            except Exception:
                logger.info("Adding conversation_summary column to user_profiles")
                await cursor.execute("ALTER TABLE user_profiles ADD COLUMN conversation_summary TEXT DEFAULT ''")

            await conn.commit()
        except Exception as e:
            logger.critical(f"Database initialization failed: {e}")

    @staticmethod
    def _read_legacy_memories_sync(file_path: str) -> List[EmotionalMemoryEntry]:
        """Synchronous helper to read legacy memories file"""
        entries = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        entries.append(EmotionalMemoryEntry.from_dict(data))
                    except Exception as e:
                        logger.warning(f"Error parsing legacy memory line: {e}")
        except Exception as e:
            logger.error(f"Error reading legacy memory file: {e}")
        return entries

    @staticmethod
    def _read_legacy_profile_sync(file_path: str) -> Optional[UserEmotionalProfile]:
        """Synchronous helper to read legacy profile file"""
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                p_data = json.load(f)
                return UserEmotionalProfile.from_dict(p_data)
        except Exception as e:
            logger.error(f"Error reading legacy profile file: {e}")
            return None

    async def _migrate_from_legacy_files(self):
        """Migrate data from legacy JSONL files if DB is empty for this user"""
        legacy_memory_file = os.path.join(self.memory_dir, f"{self.user_id}_memory.jsonl")
        legacy_profile_file = os.path.join(self.memory_dir, f"{self.user_id}_profile.json")

        if not os.path.exists(legacy_memory_file):
            return

        try:
            conn = await self._get_conn()
            cursor = await conn.cursor()
            await cursor.execute("SELECT count(*) FROM emotional_memories WHERE user_id = ?", (self.user_id,))
            row = await cursor.fetchone()
            count = row[0]

            if count > 0:
                return  # Data already exists, skip migration

            logger.info(f"Migrating legacy data for user {self.user_id}...")

            loop = asyncio.get_running_loop()

            # 1. Migrate Memories
            # File reading is sync, run in executor
            entries = await loop.run_in_executor(
                None,
                self._read_legacy_memories_sync,
                legacy_memory_file
            )

            for entry in entries:
                try:
                    await cursor.execute('''
                        INSERT INTO emotional_memories
                        (user_id, timestamp, user_input, ai_response, emotion_state, interaction_quality)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        self.user_id,
                        entry.timestamp,
                        entry.user_input,
                        entry.ai_response,
                        json.dumps(asdict(entry.emotion_state), ensure_ascii=False),
                        entry.interaction_quality
                    ))
                except Exception as e:
                    logger.warning(f"Error migrating entry: {e}")

            # 2. Migrate Profile
            if os.path.exists(legacy_profile_file):
                profile = await loop.run_in_executor(
                    None,
                    self._read_legacy_profile_sync,
                    legacy_profile_file
                )

                if profile:
                    await cursor.execute('''
                        INSERT OR REPLACE INTO user_profiles
                        (user_id, total_interactions, emotion_distribution, average_intensity,
                         common_triggers, relationship_depth, trust_level, last_interaction, emotional_patterns)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        self.user_id,
                        profile.total_interactions,
                        json.dumps(profile.emotion_distribution, ensure_ascii=False),
                        profile.average_intensity,
                        json.dumps(profile.common_triggers, ensure_ascii=False),
                        profile.relationship_depth,
                        profile.trust_level,
                        profile.last_interaction,
                        json.dumps(profile.emotional_patterns, ensure_ascii=False)
                    ))

            await conn.commit()
            logger.info("Migration completed successfully.")

            # Rename legacy files to avoid confusion (optional, keeping them for now as backup)
            # os.rename(legacy_memory_file, legacy_memory_file + ".bak")

        except Exception as e:
            logger.error(f"Migration failed: {e}")

    async def _load_profile(self) -> UserEmotionalProfile:
        """加载用户情感画像"""
        try:
            conn = await self._get_conn()
            cursor = await conn.cursor()
            await cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (self.user_id,))
            row = await cursor.fetchone()

            if row:
                # Row matches columns order in CREATE TABLE + ALTER TABLE
                # 0: user_id, 1: total, 2: dist, 3: avg_int, 4: triggers, 5: depth, 6: trust, 7: last, 8: patterns, 9: summary
                summary = ""
                if len(row) > 9:
                     summary = row[9] if row[9] else ""

                return UserEmotionalProfile(
                    user_id=row[0],
                    total_interactions=row[1],
                    emotion_distribution=json.loads(row[2]) if row[2] else {},
                    average_intensity=row[3],
                    common_triggers=json.loads(row[4]) if row[4] else [],
                    relationship_depth=row[5],
                    trust_level=row[6],
                    last_interaction=row[7] if row[7] else "",
                    emotional_patterns=json.loads(row[8]) if row[8] else {},
                    conversation_summary=summary
                )
        except Exception as e:
            logger.error(f"加载用户画像失败: {e}")

        # Return default if not found or error
        return UserEmotionalProfile(
            user_id=self.user_id,
            total_interactions=0,
            emotion_distribution={},
            average_intensity=0.0,
            common_triggers=[],
            relationship_depth=0.0,
            trust_level=0.0,
            last_interaction="",
            emotional_patterns={},
            conversation_summary=""
        )

    async def _save_profile(self):
        """保存用户情感画像"""
        try:
            conn = await self._get_conn()
            cursor = await conn.cursor()
            await cursor.execute('''
                INSERT OR REPLACE INTO user_profiles
                (user_id, total_interactions, emotion_distribution, average_intensity,
                 common_triggers, relationship_depth, trust_level, last_interaction, emotional_patterns, conversation_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.profile.user_id,
                self.profile.total_interactions,
                json.dumps(self.profile.emotion_distribution, ensure_ascii=False),
                self.profile.average_intensity,
                json.dumps(self.profile.common_triggers, ensure_ascii=False),
                self.profile.relationship_depth,
                self.profile.trust_level,
                self.profile.last_interaction,
                json.dumps(self.profile.emotional_patterns, ensure_ascii=False),
                self.profile.conversation_summary
            ))
            await conn.commit()
        except Exception as e:
            logger.error(f"保存用户画像失败: {e}")

    async def store_emotional_context(self,
                              emotion_state: EmotionState,
                              user_input: str,
                              ai_response: str) -> None:
        """
        存储情感上下文

        Args:
            emotion_state: 情感状态
            user_input: 用户输入
            ai_response: AI回应
        """
        # 内部计算交互质量
        interaction_quality = min(1.0, max(0.1, len(ai_response) / 100.0))

        entry = EmotionalMemoryEntry(
            timestamp=emotion_state.timestamp,
            emotion_state=emotion_state,
            user_input=user_input,
            ai_response=ai_response,
            interaction_quality=interaction_quality,
        )

        # 计算 embedding (用于语义检索)
        embedding_bytes = None
        try:
            embedding_fn = self._get_embedding_fn()
            # 使用 user_input 作为检索向量
            embedding_vec = embedding_fn([user_input])[0]
            # 转换为 bytes 存储
            embedding_bytes = np.array(embedding_vec, dtype=np.float32).tobytes()
            logger.debug(f"[EmotionalMemory] Computed embedding for user_input: {user_input[:20]}...")
        except Exception as e:
            logger.warning(f"[EmotionalMemory] Failed to compute embedding: {e}")

        # 保存到数据库
        try:
            conn = await self._get_conn()
            cursor = await conn.cursor()
            await cursor.execute('''
                INSERT INTO emotional_memories
                (user_id, timestamp, user_input, ai_response, emotion_state, interaction_quality, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.user_id,
                entry.timestamp,
                entry.user_input,
                entry.ai_response,
                json.dumps(asdict(entry.emotion_state), ensure_ascii=False),
                entry.interaction_quality,
                embedding_bytes
            ))
            await conn.commit()
        except Exception as e:
            logger.error(f"保存情感记忆失败: {e}")

        # 更新内存中的 profile 并保存
        await self._update_profile(entry)

    async def _update_profile(self, entry: EmotionalMemoryEntry):
        """更新用户情感画像（内存中）"""
        emotion = entry.emotion_state.primary_emotion
        intensity = entry.emotion_state.intensity
        triggers = entry.emotion_state.triggers

        self.profile.total_interactions += 1
        self.profile.last_interaction = entry.timestamp

        if emotion not in self.profile.emotion_distribution:
            self.profile.emotion_distribution[emotion] = 0
        self.profile.emotion_distribution[emotion] += 1

        total_intensity = self.profile.average_intensity * (self.profile.total_interactions - 1) + intensity
        self.profile.average_intensity = total_intensity / self.profile.total_interactions

        for trigger in triggers:
            if trigger not in self.profile.common_triggers:
                self.profile.common_triggers.append(trigger)

        self._update_relationship_metrics(entry)
        self._update_emotional_patterns(entry)

        # 标记为脏数据，计数器+1
        self._profile_dirty = True
        self._update_counter += 1

        # 达到阈值时自动写入数据库
        if self._update_counter >= self._flush_threshold:
            await self._flush_profile()

    async def _flush_profile(self):
        """强制写入profile到数据库"""
        if not self._profile_dirty:
            return

        await self._save_profile()
        self._update_counter = 0
        self._profile_dirty = False
        logger.debug("[EmotionalMemory] Profile flushed to DB")

    def _update_relationship_metrics(self, entry: EmotionalMemoryEntry):
        """
        更新关系深度和信任度（优化版本）
        优化：考虑情感类型、互动频率、负面情绪响应质量
        """
        quality_factor = entry.interaction_quality
        intensity = entry.emotion_state.intensity
        emotion = entry.emotion_state.primary_emotion

        # 情感类型权重：积极情感促进关系
        positive_emotions = ["开心", "平静"]
        negative_emotions = ["难过", "焦虑", "孤独", "愤怒", "疲惫"]

        emotion_bonus = 1.0
        if emotion in positive_emotions:
            emotion_bonus = 1.2  # 积极情感加分
        elif emotion in negative_emotions:
            emotion_bonus = 0.9  # 消极情感略微减分（需要更多努力）

        depth_factor = min(intensity * 1.2, 1.0)

        # 简单处理时间解析
        try:
            days_ago = (datetime.now() - datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S")).days
        except:
            days_ago = 0

        # 互动频率因子：最近互动越频繁，关系增长越快
        freq_factor = 1.0
        if self.profile.last_interaction:
            try:
                last_time = datetime.strptime(self.profile.last_interaction, "%Y-%m-%d %H:%M:%S")
                hours_since_last = (datetime.now() - last_time).total_seconds() / 3600
                if hours_since_last < 2:  # 2小时内再次互动
                    freq_factor = 1.3
                elif hours_since_last < 24:  # 24小时内
                    freq_factor = 1.1
                elif hours_since_last > 168:  # 一周没互动
                    freq_factor = 0.7
            except:
                pass

        time_factor = max(0.1, 1.0 - (days_ago / 30))

        # 衰减因子：关系深度越高，增长越慢
        depth_decay = 1.0 - self.profile.relationship_depth * 0.5
        trust_decay = 1.0 - self.profile.trust_level * 0.5

        # 更新关系深度：综合考虑多个因素
        relationship_increment = (
            quality_factor * depth_factor * time_factor * depth_decay
            * emotion_bonus * freq_factor
        ) * 0.05
        self.profile.relationship_depth = min(1.0, self.profile.relationship_depth + relationship_increment)

        # 更新信任度
        if quality_factor > 0.7:
            # 高质量回应增加信任
            trust_increment = 0.02 * time_factor * trust_decay * emotion_bonus
            self.profile.trust_level = min(1.0, self.profile.trust_level + trust_increment)
        elif quality_factor < 0.3:
            # 低质量回应减少信任
            trust_decrement = 0.01
            self.profile.trust_level = max(0.0, self.profile.trust_level - trust_decrement)

    def _update_emotional_patterns(self, entry: EmotionalMemoryEntry):
        """更新情感模式分析"""
        try:
            hour = str(datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S").hour)
        except:
            hour = str(datetime.now().hour)

        emotion = entry.emotion_state.primary_emotion

        if "time_patterns" not in self.profile.emotional_patterns:
            self.profile.emotional_patterns["time_patterns"] = {}

        if hour not in self.profile.emotional_patterns["time_patterns"]:
            self.profile.emotional_patterns["time_patterns"][hour] = {}

        if emotion not in self.profile.emotional_patterns["time_patterns"][hour]:
            self.profile.emotional_patterns["time_patterns"][hour][emotion] = 0

        self.profile.emotional_patterns["time_patterns"][hour][emotion] += 1

        if "trigger_patterns" not in self.profile.emotional_patterns:
            self.profile.emotional_patterns["trigger_patterns"] = {}

        for trigger in entry.emotion_state.triggers:
            if trigger not in self.profile.emotional_patterns["trigger_patterns"]:
                self.profile.emotional_patterns["trigger_patterns"][trigger] = {}

            if emotion not in self.profile.emotional_patterns["trigger_patterns"][trigger]:
                self.profile.emotional_patterns["trigger_patterns"][trigger][emotion] = 0

            self.profile.emotional_patterns["trigger_patterns"][trigger][emotion] += 1

    async def get_emotional_history(self, days: int = 7) -> List[EmotionalMemoryEntry]:
        """获取指定天数内的情感历程"""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        entries = []

        try:
            conn = await self._get_conn()
            cursor = await conn.cursor()
            await cursor.execute('''
                SELECT * FROM emotional_memories
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (self.user_id, cutoff_date))

            rows = await cursor.fetchall()
            for row in rows:
                try:
                    # row: 0:id, 1:user_id, 2:timestamp, 3:input, 4:response, 5:emotion_json, 6:quality
                    emotion_dict = json.loads(row[5])
                    # Use dict to create EmotionState
                    valid_fields = {"primary_emotion", "intensity", "confidence", "context", "triggers", "timestamp"}
                    filtered_emotion = {k: v for k, v in emotion_dict.items() if k in valid_fields}
                    emotion_state = EmotionState(**filtered_emotion)

                    entry = EmotionalMemoryEntry(
                        timestamp=row[2],
                        emotion_state=emotion_state,
                        user_input=row[3],
                        ai_response=row[4],
                        interaction_quality=row[6]
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"解析记忆条目失败: {e}")

        except Exception as e:
            logger.error(f"读取情感记忆失败: {e}")

        return entries

    async def get_relevant_memories(self, query: str, limit: int = 5, days: int = 30) -> List[EmotionalMemoryEntry]:
        """
        基于语义相似度检索相关情感记忆

        Args:
            query: 用户当前输入（用于计算相似度）
            limit: 返回数量限制
            days: 检索天数范围

        Returns:
            List[EmotionalMemoryEntry]: 按语义相似度排序的记忆列表
        """
        # 计算查询的 embedding
        query_embedding = None
        try:
            embedding_fn = self._get_embedding_fn()
            query_embedding = embedding_fn([query])[0]
        except Exception as e:
            logger.warning(f"[EmotionalMemory] Failed to compute query embedding: {e}")
            # 回退到时间检索
            return await self.get_emotional_history(days=days)

        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

        try:
            conn = await self._get_conn()
            cursor = await conn.cursor()
            # 获取所有有 embedding 的记忆
            await cursor.execute('''
                SELECT id, user_id, timestamp, user_input, ai_response, emotion_state, interaction_quality, embedding
                FROM emotional_memories
                WHERE user_id = ? AND timestamp >= ? AND embedding IS NOT NULL
            ''', (self.user_id, cutoff_date))

            rows = await cursor.fetchall()

            if not rows:
                logger.info("[EmotionalMemory] No memories with embeddings found, falling back to time-based")
                return await self.get_emotional_history(days=days)

            # 计算余弦相似度
            similarities = []
            query_vec = np.array(query_embedding, dtype=np.float32)

            for row in rows:
                try:
                    embedding_bytes = row[7]
                    if embedding_bytes is None:
                        continue

                    mem_vec = np.frombuffer(embedding_bytes, dtype=np.float32)
                    # 余弦相似度 (因为向量已经 normalize，所以直接点积)
                    similarity = np.dot(query_vec, mem_vec)

                    similarities.append((similarity, row))
                except Exception as e:
                    logger.warning(f"[EmotionalMemory] Failed to compute similarity: {e}")
                    continue

            # 按相似度排序
            similarities.sort(key=lambda x: x[0], reverse=True)

            # 取 top-k
            top_results = similarities[:limit]

            entries = []
            for similarity, row in top_results:
                try:
                    emotion_dict = json.loads(row[5])
                    valid_fields = {"primary_emotion", "intensity", "confidence", "context", "triggers", "timestamp"}
                    filtered_emotion = {k: v for k, v in emotion_dict.items() if k in valid_fields}
                    emotion_state = EmotionState(**filtered_emotion)

                    entry = EmotionalMemoryEntry(
                        timestamp=row[2],
                        emotion_state=emotion_state,
                        user_input=row[3],
                        ai_response=row[4],
                        interaction_quality=row[6]
                    )
                    entry.similarity = float(similarity)  # 添加相似度字段
                    entries.append(entry)
                    logger.info(f"[EmotionalMemory] Found relevant memory (sim={similarity:.3f}): {row[3][:30]}...")
                except Exception as e:
                    logger.warning(f"[EmotionalMemory] Failed to parse memory entry: {e}")
                    continue

            return entries

        except Exception as e:
            logger.error(f"语义检索情感记忆失败: {e}")
            return await self.get_emotional_history(days=days)

    def get_profile_summary(self) -> Dict[str, Any]:
        """获取用户画像摘要"""
        return {
            "user_id": self.profile.user_id,
            "total_interactions": self.profile.total_interactions,
            "dominant_emotions": sorted(self.profile.emotion_distribution.items(), key=lambda x: x[1], reverse=True)[:3],
            "average_intensity": self.profile.average_intensity,
            "common_triggers": self.profile.common_triggers[:5],
            "relationship_depth": self.profile.relationship_depth,
            "trust_level": self.profile.trust_level,
            "last_interaction": self.profile.last_interaction
        }
