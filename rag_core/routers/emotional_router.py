"""
情感路由器 - Emotional Router
负责识别用户情感状态并判断是否为纯情感倾诉
优化版本：添加快速路径，减少不必要的LLM调用
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from rag_core.llm.llm_client import LLMClient
from rag_core.utils.logger import logger


def load_emotion_keywords() -> dict:
    """从 JSON 配置文件加载情感词典"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "emotion_keywords.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# 快速路径置信度阈值
KEYWORD_HIGH_CONFIDENCE_THRESHOLD = 0.8
KEYWORD_MEDIUM_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class EmotionState:
    """情感状态数据结构"""
    primary_emotion: str  # 主要情感：开心、难过、焦虑、兴奋、孤独、愤怒等
    intensity: float  # 强度：0.0-1.0
    confidence: float  # 置信度：0.0-1.0
    context: str  # 情感上下文
    triggers: List[str]  # 情感触发因素
    timestamp: str  # 时间戳


class EmotionalRouter:
    def __init__(self):
        self.client = LLMClient.get_instance()
        self._init_emotion_lexicon()

    def _init_emotion_lexicon(self):
        """从配置文件加载情感词典"""
        emotion_config = load_emotion_keywords()
        self.emotion_keywords = emotion_config["emotion_keywords"]
        self.intensity_indicators = emotion_config["intensity_indicators"]
        self._pure_emotional_phrases = emotion_config["pure_emotional_phrases"]

    async def analyze_emotion(self, user_input: str, history: Optional[List[Dict]] = None) -> EmotionState:
        """
        分析用户输入的情感状态
        优化：添加快速路径，关键词检测置信度高时跳过LLM调用

        Args:
            user_input: 用户输入
            history: 对话历史（可选）

        Returns:
            EmotionState: 情感状态对象
        """
        # 1. 基于关键词的快速情感检测
        keyword_result = self._detect_emotion_by_keywords(user_input)

        # 快速路径：关键词检测置信度非常高，直接返回
        if keyword_result.confidence >= KEYWORD_HIGH_CONFIDENCE_THRESHOLD:
            logger.debug(f"[EmotionalRouter] 快速路径: 关键词置信度 {keyword_result.confidence:.2f} >= {KEYWORD_HIGH_CONFIDENCE_THRESHOLD}, 跳过LLM")
            return keyword_result

        # 2. 检查是否在纯情感倾诉短语列表中
        if any(phrase in user_input for phrase in self._pure_emotional_phrases):
            logger.debug(f"[EmotionalRouter] 快速路径: 检测到纯情感倾诉短语")
            return keyword_result

        # 3. 常规路径：使用LLM进行深度情感分析
        llm_result = await self._analyze_emotion_with_llm(user_input, history if history else [])

        # 4. 选择最佳结果
        if llm_result.confidence >= KEYWORD_MEDIUM_CONFIDENCE_THRESHOLD:
            return llm_result
        return keyword_result

    def _detect_emotion_by_keywords(self, user_input: str) -> EmotionState:
        """基于关键词的情感检测（快速方法）"""
        text = user_input.lower()

        # 检测主要情感
        detected_emotion = "平静"
        max_score = 0
        total_matches = 0

        for emotion, keywords in self.emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            total_matches += matches
            if matches > max_score:
                max_score = matches
                detected_emotion = emotion

        # 如果没有匹配任何关键词，返回平静
        if max_score == 0:
            return EmotionState(
                primary_emotion="平静",
                intensity=0.3,
                confidence=0.5,
                context=user_input,
                triggers=["日常"],
                timestamp=self._get_timestamp()
            )

        # 计算置信度：基于匹配数量
        # 单个强匹配 = 高置信度，多个弱匹配 = 中等置信度
        if max_score >= 2:
            confidence = 0.85  # 强匹配
        elif max_score == 1:
            confidence = 0.7   # 中等匹配
        else:
            confidence = 0.5

        # 检测强度
        intensity = 0.5  # 默认中等强度
        for level, indicators in self.intensity_indicators.items():
            if any(indicator in text for indicator in indicators):
                if level == "high":
                    intensity = 0.85
                elif level == "medium":
                    intensity = 0.6
                elif level == "low":
                    intensity = 0.35
                break

        # 提取触发因素
        triggers = self._extract_triggers(text)

        return EmotionState(
            primary_emotion=detected_emotion,
            intensity=intensity,
            confidence=confidence,
            context=user_input,
            triggers=triggers,
            timestamp=self._get_timestamp()
        )

    async def _analyze_emotion_with_llm(self, user_input: str, history: Optional[List[Dict]] = None) -> EmotionState:
        """使用LLM进行深度情感分析"""
        context_str = "无"
        if history and len(history) > 0:
            recent_turns = history[-6:] if len(history) >= 6 else history
            context_str = json.dumps(recent_turns, ensure_ascii=False)

        system_prompt = (
            "你是情感分析专家。请分析用户的情感状态，只返回JSON格式的结果。\n"
            "分析维度包括：\n"
            "1. primary_emotion: 主要情感（开心/难过/焦虑/孤独/愤怒/疲惫/困惑/平静）\n"
            "2. intensity: 情感强度（0.0-1.0）\n"
            "3. confidence: 分析置信度（0.0-1.0）\n"
            "4. triggers: 情感触发因素（字符串列表）\n"
            "5. context: 情感上下文描述\n\n"
            "示例输入：'今天工作好累啊，压力好大，感觉什么都不顺'\n"
            "示例输出：{\"primary_emotion\": \"疲惫\", \"intensity\": 0.7, \"confidence\": 0.8, "
            "\"triggers\": [\"工作压力\", \"事情不顺利\"], \"context\": \"因工作压力和挫折感到疲惫\"}\n\n"
            f"对话历史上下文：{context_str}\n\n"
            "请分析以下用户输入："
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]

            response = await self.client.client.chat.completions.create(
                model=self.client.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty content")

            result = json.loads(content)

            return EmotionState(
                primary_emotion=result.get("primary_emotion", "平静"),
                intensity=float(result.get("intensity", 0.5)),
                confidence=float(result.get("confidence", 0.5)),
                context=result.get("context", user_input),
                triggers=result.get("triggers", []),
                timestamp=self._get_timestamp()
            )

        except Exception as e:
            logger.error(f"[EmotionalRouter] LLM分析失败: {e}")
            return self._detect_emotion_by_keywords(user_input)

    def _extract_triggers(self, text: str) -> List[str]:
        """提取情感触发因素"""
        triggers = []

        trigger_patterns = {
            "工作": ["工作", "上班", "公司", "老板", "同事", "加班", "项目", "任务"],
            "学习": ["学习", "考试", "作业", "学校", "老师", "同学", "功课"],
            "感情": ["恋爱", "分手", "喜欢", "爱情", "男/女朋友", "暧昧"],
            "家庭": ["家人", "父母", "家", "家庭", "兄弟", "姐妹"],
            "健康": ["生病", "不舒服", "健康", "身体", "医院", "药"],
            "经济": ["钱", "经济", "财务", "工资", "收入", "花费", "穷"]
        }

        for trigger, keywords in trigger_patterns.items():
            if any(keyword in text for keyword in keywords):
                triggers.append(trigger)

        return triggers if triggers else ["日常"]

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def is_pure_emotional_query(self, user_input: str, emotion_state: EmotionState) -> bool:
        """
        判断是否为纯情感倾诉（大幅收紧条件）。
        只有明确的纯情感倾诉才返回True，默认返回False让IntentRouter决定是否需要工具。

        Args:
            user_input: 用户输入
            emotion_state: 情感状态

        Returns:
            bool: 是否为纯情感倾诉
        """
        text = user_input.strip()

        # 1. 明确的纯情感倾诉短语
        if any(phrase in text for phrase in self._pure_emotional_phrases):
            return True

        # 2. 高强度负面情感 + 短文本（无查询意图）
        negative_emotions = ["难过", "焦虑", "孤独", "愤怒", "疲惫"]
        if (emotion_state.primary_emotion in negative_emotions
                and emotion_state.intensity >= 0.7
                and len(text) <= 20):
            # 排除包含查询意图的情况
            query_indicators = ["什么", "怎么", "哪", "谁", "为什么", "多少",
                                "告诉我", "给我", "查", "搜", "找", "讲讲", "介绍"]
            if not any(indicator in text for indicator in query_indicators):
                return True

        return False
