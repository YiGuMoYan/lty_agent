"""
情感记忆系统 - Emotional Memory
负责存储、管理和检索用户的情感历程，建立长期情感档案
"""

import json
import os
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
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            memory_dir = os.path.join(base_dir, "dataset", "emotional_memory")

        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        self.memory_file = os.path.join(memory_dir, f"{user_id}_memory.jsonl")
        self.profile_file = os.path.join(memory_dir, f"{user_id}_profile.json")

        self.profile = self._load_profile()

    def _load_profile(self) -> UserEmotionalProfile:
        """加载用户情感画像"""
        try:
            if os.path.exists(self.profile_file):
                with open(self.profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return UserEmotionalProfile.from_dict(data)
        except Exception as e:
            print(f"[EmotionalMemory] 加载用户画像失败: {e}")

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
            with open(self.profile_file, 'w', encoding='utf-8') as f:
                json.dump(self.profile.to_dict(), f, ensure_ascii=False, indent=2)
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

        # 保存到文件
        try:
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[EmotionalMemory] 保存情感记忆失败: {e}")

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

        days_ago = (datetime.now() - datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S")).days
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
        hour = datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S").hour
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
        cutoff_date = datetime.now() - timedelta(days=days)
        entries = []

        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                entry = EmotionalMemoryEntry.from_dict(data)
                                entry_date = datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S")
                                if entry_date >= cutoff_date:
                                    entries.append(entry)
                            except Exception as e:
                                print(f"[EmotionalMemory] 解析记忆条目失败: {e}")
        except Exception as e:
            print(f"[EmotionalMemory] 读取情感记忆失败: {e}")

        entries.sort(key=lambda x: x.timestamp, reverse=True)
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
