"""
回复风格管理模块 - Response Style Manager
负责管理对话回复的风格配置，支持多种风格选项
"""

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass


class ResponseStyle(str, Enum):
    """回复风格枚举"""
    CASUAL = "casual"  # 口语化：日常交流风格，柔和自然
    PROFESSIONAL = "professional"  # 专业：知识查询风格，准确清晰
    CONCISE = "concise"  # 简短：高效交流风格，简洁明了


@dataclass
class StyleConfig:
    """风格配置数据结构"""
    style: ResponseStyle
    description: str
    max_response_length: int  # 最大回复长度（句数）


# 风格配置映射表
STYLE_CONFIGS: Dict[ResponseStyle, StyleConfig] = {
    ResponseStyle.CASUAL: StyleConfig(
        style=ResponseStyle.CASUAL,
        description="口语化：日常交流风格，柔和自然，适合情感陪伴",
        max_response_length=5,
    ),

    ResponseStyle.PROFESSIONAL: StyleConfig(
        style=ResponseStyle.PROFESSIONAL,
        description="专业：知识查询风格，准确清晰，适合信息检索",
        max_response_length=10,
    ),

    ResponseStyle.CONCISE: StyleConfig(
        style=ResponseStyle.CONCISE,
        description="简短：高效交流风格，简洁明了，适合快速问答",
        max_response_length=3,
    )
}


class StyleManager:
    """风格管理器"""

    def __init__(self, default_style: ResponseStyle = ResponseStyle.CASUAL):
        """
        初始化风格管理器

        Args:
            default_style: 默认风格
        """
        self.current_style = default_style
        self.default_style = default_style

    def set_style(self, style: ResponseStyle) -> bool:
        """
        设置当前风格

        Args:
            style: 目标风格

        Returns:
            bool: 设置是否成功
        """
        if style in STYLE_CONFIGS:
            self.current_style = style
            return True
        return False

    def get_current_style(self) -> ResponseStyle:
        """获取当前风格"""
        return self.current_style

    def get_style_config(self, style: Optional[ResponseStyle] = None) -> StyleConfig:
        """
        获取风格配置

        Args:
            style: 风格类型（None 表示使用当前风格）

        Returns:
            StyleConfig: 风格配置
        """
        if style is None:
            style = self.current_style
        return STYLE_CONFIGS.get(style, STYLE_CONFIGS[ResponseStyle.CASUAL])

    def reset_to_default(self) -> None:
        """重置为默认风格"""
        self.current_style = self.default_style

    def get_available_styles(self) -> Dict[str, str]:
        """
        获取所有可用风格的描述

        Returns:
            Dict[str, str]: 风格名称到描述的映射
        """
        return {
            style.value: config.description
            for style, config in STYLE_CONFIGS.items()
        }

    def get_max_response_length(self) -> int:
        """获取当前风格的最大回复长度"""
        return self.get_style_config().max_response_length


def parse_style_from_string(style_str: str) -> ResponseStyle:
    """
    从字符串解析风格

    Args:
        style_str: 风格字符串（casual/professional/concise）

    Returns:
        ResponseStyle: 风格枚举

    Raises:
        ValueError: 无效的风格字符串
    """
    try:
        return ResponseStyle(style_str.lower())
    except ValueError:
        raise ValueError(f"Invalid style: {style_str}. Available styles: {', '.join([s.value for s in ResponseStyle])}")


def get_style_description(style: ResponseStyle) -> str:
    """
    获取风格的描述

    Args:
        style: 风格类型

    Returns:
        str: 风格描述
    """
    config = STYLE_CONFIGS.get(style)
    return config.description if config else "未知风格"
