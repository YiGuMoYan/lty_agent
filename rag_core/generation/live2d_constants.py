# -*- coding: utf-8 -*-
"""
Live2D 公共常量定义
从 unified_generator.py 和 live2d_generator.py 抽取的公共常量
"""

# Live2D参数范围定义
PARAM_RANGES = {
    "ParamAngleX": (-30, 30), "ParamAngleY": (-30, 30), "ParamAngleZ": (-30, 30),
    "ParamEyeLOpen": (0, 1), "ParamEyeROpen": (0, 1),
    "ParamEyeLSmile": (0, 1), "ParamEyeRSmile": (0, 1),
    "ParamEyeBallX": (-1, 1), "ParamEyeBallY": (-1, 1),
    "ParamBrowLY": (-1, 1), "ParamBrowRY": (-1, 1),
    "ParamBrowLAngle": (-1, 1), "ParamBrowRAngle": (-1, 1),
    "ParamMouthForm": (-1, 1), "ParamMouthOpenY": (0, 1),
    "ParamCheek": (0, 1),
    "ParamBodyAngleX": (-10, 10), "ParamBodyAngleY": (-10, 10), "ParamBodyAngleZ": (-10, 10),
    "ParamBreath": (0, 1),
}

# 有效的姿势列表
VALID_POSES = {
    "ParamPOSE1", "ParamPOSE2", "ParamPOSE3", "ParamPOSE4", "ParamPOSE5",
    "ParamPOSE6", "ParamPOSE7", "ParamPOSE8", "ParamPOSE10"
}

# 必需的参数列表（用于验证）
REQUIRED_PARAMS = [
    "ParamEyeLOpen", "ParamEyeROpen",
    "ParamEyeLSmile", "ParamEyeRSmile",
    "ParamEyeBallX", "ParamEyeBallY",
    "ParamBrowLY", "ParamBrowRY",
    "ParamBrowLAngle", "ParamBrowRAngle",
    "ParamMouthForm", "ParamMouthOpenY"
]

# 中性基线参数（情绪映射的起点，也作为默认值）
# 注意：此字典同时用于情绪插值和 LLM 未生成参数时的备用
NEUTRAL_PARAMS = {
    "ParamAngleX": 0,
    "ParamAngleY": 0,
    "ParamAngleZ": 0,
    "ParamEyeLOpen": 1,
    "ParamEyeROpen": 1,
    "ParamEyeLSmile": 0,
    "ParamEyeRSmile": 0,
    "ParamEyeBallX": 0,
    "ParamEyeBallY": 0,
    "ParamBrowLY": 0,
    "ParamBrowRY": 0,
    "ParamBrowLAngle": 0,
    "ParamBrowRAngle": 0,
    "ParamMouthForm": 0,
    "ParamMouthOpenY": 0,
    "ParamCheek": 0,
    "ParamBodyAngleX": 0,
    "ParamBodyAngleY": 0,
    "ParamBodyAngleZ": 0,
    "ParamBreath": 0,
}

# 参数默认值（当LLM未生成时使用），使用与 NEUTRAL_PARAMS 相同的值
DEFAULT_PARAMS = NEUTRAL_PARAMS.copy()


def clamp_param(param_name: str, value: float) -> float:
    """将参数裁剪到合法范围"""
    if param_name not in PARAM_RANGES:
        return value
    lo, hi = PARAM_RANGES[param_name]
    return max(lo, min(hi, value))


def validate_pose(pose: str) -> str:
    """验证姿势是否有效"""
    if pose and pose in VALID_POSES:
        return pose
    return None


def fill_missing_params(params: dict) -> dict:
    """填充缺失的参数到默认值"""
    result = DEFAULT_PARAMS.copy()
    if params:
        for k, v in params.items():
            if k in PARAM_RANGES:
                result[k] = clamp_param(k, v)
    return result
