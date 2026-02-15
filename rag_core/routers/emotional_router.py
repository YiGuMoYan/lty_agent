"""
情感路由器 - Emotional Router
负责识别用户情感状态并判断是否为纯情感倾诉
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from rag_core.llm.llm_client import LLMClient


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
        self.client = LLMClient()
        self._init_emotion_lexicon()

    def _init_emotion_lexicon(self):
        """初始化情感词典"""
        self.emotion_keywords = {
            "开心": ["开心", "快乐", "高兴", "兴奋", "愉快", "欢乐", "喜悦", "满足", "幸福", "乐", "哈哈", "嘻嘻", "哇"],
            "难过": ["难过", "伤心", "悲伤", "沮丧", "失落", "郁闷", "痛苦", "哭", "眼泪", "呜呜", "唉"],
            "焦虑": ["焦虑", "紧张", "担心", "害怕", "恐惧", "不安", "压力", "烦躁", "忐忑", "慌", "怕"],
            "孤独": ["孤独", "寂寞", "一个人", "孤单", "没人", "冷清", "无聊", "空虚"],
            "愤怒": ["生气", "愤怒", "恼火", "气愤", "火大", "不爽", "烦", "气", "恨"],
            "疲惫": ["累", "疲惫", "疲倦", "困", "疲劳", "没精神", "乏力", "筋疲力尽"],
            "困惑": ["困惑", "迷茫", "不知道", "不明白", "疑惑", "疑问", "糊涂", "搞不懂"],
        }

        self.intensity_indicators = {
            "high": ["非常", "特别", "超级", "极其", "真的", "太", "超", "爆", "绝了", "max"],
            "medium": ["比较", "还算", "挺", "蛮", "略微"],
            "low": ["一点", "稍微", "有点", "不太", "不怎么"]
        }

        # 纯情感倾诉表达
        self._pure_emotional_phrases = [
            "陪陪我", "好难受", "想哭", "受不了了", "好想哭",
            "心好累", "好烦", "好孤独", "好寂寞", "抱抱我",
            "安慰我", "我好难过", "我好累", "我好烦",
        ]

    async def analyze_emotion(self, user_input: str, history: Optional[List[Dict]] = None) -> EmotionState:
        """
        分析用户输入的情感状态

        Args:
            user_input: 用户输入
            history: 对话历史（可选）

        Returns:
            EmotionState: 情感状态对象
        """
        # 1. 基于关键词的快速情感检测
        keyword_result = self._detect_emotion_by_keywords(user_input)

        # 2. 使用LLM进行深度情感分析
        llm_result = await self._analyze_emotion_with_llm(user_input, history if history else [])

        # 3. 选择最佳结果
        if llm_result.confidence > 0.6:
            return llm_result
        return keyword_result

    def _detect_emotion_by_keywords(self, user_input: str) -> EmotionState:
        """基于关键词的情感检测（快速方法）"""
        text = user_input.lower()

        # 检测主要情感
        detected_emotion = "平静"
        max_score = 0

        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > max_score:
                max_score = score
                detected_emotion = emotion

        # 检测强度
        intensity = 0.5  # 默认中等强度
        for level, indicators in self.intensity_indicators.items():
            if any(indicator in text for indicator in indicators):
                if level == "high":
                    intensity = 0.8
                elif level == "medium":
                    intensity = 0.6
                elif level == "low":
                    intensity = 0.3
                break

        # 提取触发因素
        triggers = self._extract_triggers(text)

        return EmotionState(
            primary_emotion=detected_emotion,
            intensity=intensity,
            confidence=0.6,  # 关键词法置信度中等
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
            print(f"[EmotionalRouter] LLM分析失败: {e}")
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
