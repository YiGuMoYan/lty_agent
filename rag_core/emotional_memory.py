"""
情感记忆系统 - Emotional Memory
负责存储、管理和检索用户的情感历程，建立长期情感档案
Refactored to use SQLite for scalability.
"""

import json
import os
import sqlite3
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from .emotional_router import EmotionState


@dataclass
class EmotionalMemoryEntry:
    """情感记忆条目"""
    timestamp: str
    emotion_state: EmotionState
    user_input: str
    ai_response: str
    interaction_quality: float  # 交互质量评分 0.0-1.0

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp,
            "emotion_state": asdict(self.emotion_state),
            "user_input": self.user_input,
            "ai_response": self.ai_response,
            "interaction_quality": self.interaction_quality,
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
            # rag_core/emotional_memory.py -> rag_core -> rag_lty (Root)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            memory_dir = os.path.join(base_dir, "dataset", "emotional_memory")

        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        self.db_path = os.path.join(memory_dir, "emotional_memory.db")
        self._init_db()

        # Check for legacy file migration
        self._migrate_from_legacy_files()

        self.profile = self._load_profile()

    def _get_conn(self):
        """Get SQLite connection"""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        """Initialize database schema"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                # User Profiles Table
                cursor.execute('''
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
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS emotional_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_input TEXT,
                        ai_response TEXT,
                        emotion_state TEXT,
                        interaction_quality REAL,
                        FOREIGN KEY(user_id) REFERENCES user_profiles(user_id)
                    )
                ''')
                # Index for faster history retrieval
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_user_timestamp ON emotional_memories(user_id, timestamp)')
                conn.commit()
        except Exception as e:
            print(f"[EmotionalMemory] Database initialization failed: {e}")

    def _migrate_from_legacy_files(self):
        """Migrate data from legacy JSONL files if DB is empty for this user"""
        legacy_memory_file = os.path.join(self.memory_dir, f"{self.user_id}_memory.jsonl")
        legacy_profile_file = os.path.join(self.memory_dir, f"{self.user_id}_profile.json")

        if not os.path.exists(legacy_memory_file):
            return

        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM emotional_memories WHERE user_id = ?", (self.user_id,))
                count = cursor.fetchone()[0]

                if count > 0:
                    return  # Data already exists, skip migration

                print(f"[EmotionalMemory] Migrating legacy data for user {self.user_id}...")

                # 1. Migrate Memories
                with open(legacy_memory_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            data = json.loads(line)
                            entry = EmotionalMemoryEntry.from_dict(data)
                            cursor.execute('''
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
                            print(f"[EmotionalMemory] Error migrating line: {e}")

                # 2. Migrate Profile
                if os.path.exists(legacy_profile_file):
                    with open(legacy_profile_file, 'r', encoding='utf-8') as f:
                        p_data = json.load(f)
                        profile = UserEmotionalProfile.from_dict(p_data)
                        cursor.execute('''
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

                conn.commit()
                print(f"[EmotionalMemory] Migration completed successfully.")

                # Rename legacy files to avoid confusion (optional, keeping them for now as backup)
                # os.rename(legacy_memory_file, legacy_memory_file + ".bak")

        except Exception as e:
            print(f"[EmotionalMemory] Migration failed: {e}")

    def _load_profile(self) -> UserEmotionalProfile:
        """加载用户情感画像"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (self.user_id,))
                row = cursor.fetchone()

                if row:
                    # Row matches columns order in CREATE TABLE
                    # 0: user_id, 1: total, 2: dist, 3: avg_int, 4: triggers, 5: depth, 6: trust, 7: last, 8: patterns
                    return UserEmotionalProfile(
                        user_id=row[0],
                        total_interactions=row[1],
                        emotion_distribution=json.loads(row[2]) if row[2] else {},
                        average_intensity=row[3],
                        common_triggers=json.loads(row[4]) if row[4] else [],
                        relationship_depth=row[5],
                        trust_level=row[6],
                        last_interaction=row[7] if row[7] else "",
                        emotional_patterns=json.loads(row[8]) if row[8] else {}
                    )
        except Exception as e:
            print(f"[EmotionalMemory] 加载用户画像失败: {e}")

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
            emotional_patterns={}
        )

    def _save_profile(self):
        """保存用户情感画像"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_profiles
                    (user_id, total_interactions, emotion_distribution, average_intensity,
                     common_triggers, relationship_depth, trust_level, last_interaction, emotional_patterns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.profile.user_id,
                    self.profile.total_interactions,
                    json.dumps(self.profile.emotion_distribution, ensure_ascii=False),
                    self.profile.average_intensity,
                    json.dumps(self.profile.common_triggers, ensure_ascii=False),
                    self.profile.relationship_depth,
                    self.profile.trust_level,
                    self.profile.last_interaction,
                    json.dumps(self.profile.emotional_patterns, ensure_ascii=False)
                ))
                conn.commit()
        except Exception as e:
            print(f"[EmotionalMemory] 保存用户画像失败: {e}")

    def store_emotional_context(self,
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

        # 保存到数据库
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
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
                conn.commit()
        except Exception as e:
            print(f"[EmotionalMemory] 保存情感记忆失败: {e}")

        # 更新内存中的 profile 并保存
        self._update_profile(entry)

    def _update_profile(self, entry: EmotionalMemoryEntry):
        """更新用户情感画像"""
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
        self._save_profile()

    def _update_relationship_metrics(self, entry: EmotionalMemoryEntry):
        """更新关系深度和信任度（含衰减因子）"""
        quality_factor = entry.interaction_quality
        depth_factor = min(entry.emotion_state.intensity * 1.2, 1.0)

        # 简单处理时间解析
        try:
            days_ago = (datetime.now() - datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S")).days
        except:
            days_ago = 0

        time_factor = max(0.1, 1.0 - (days_ago / 30))

        # 衰减因子：关系深度越高，增长越慢
        depth_decay = 1.0 - self.profile.relationship_depth * 0.5
        trust_decay = 1.0 - self.profile.trust_level * 0.5

        # 更新关系深度
        relationship_increment = (quality_factor * depth_factor * time_factor * depth_decay) * 0.05
        self.profile.relationship_depth = min(1.0, self.profile.relationship_depth + relationship_increment)

        # 更新信任度
        if quality_factor > 0.7:
            trust_increment = 0.02 * time_factor * trust_decay
            self.profile.trust_level = min(1.0, self.profile.trust_level + trust_increment)
        elif quality_factor < 0.3:
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

    def get_emotional_history(self, days: int = 7) -> List[EmotionalMemoryEntry]:
        """获取指定天数内的情感历程"""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        entries = []

        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM emotional_memories
                    WHERE user_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (self.user_id, cutoff_date))

                rows = cursor.fetchall()
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
                        print(f"[EmotionalMemory] 解析记忆条目失败: {e}")

        except Exception as e:
            print(f"[EmotionalMemory] 读取情感记忆失败: {e}")

        return entries

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
