"""
情绪 → Live2D 参数映射
将 EmotionState 的 primary_emotion + intensity 转换为 Live2D 模型参数字典
"""

# 从 constants 导入中性基线参数
from rag_core.generation.live2d_constants import NEUTRAL_PARAMS

# 每种情绪在 intensity=1.0 时的目标参数（只列出与中性不同的）
# 眼球始终看前方（不设置 EyeBallX/Y），表情幅度加大
EMOTION_PARAMS = {
    "开心": {
        "ParamEyeLSmile": 1.0,
        "ParamEyeRSmile": 1.0,
        "ParamMouthForm": 1.0,
        "ParamMouthOpenY": 0.5,
        "ParamCheek": 1.0,
        "ParamAngleZ": 10,
        "ParamBrowLY": 0.6,
        "ParamBrowRY": 0.6,
        "ParamBodyAngleZ": 4,
    },
    "难过": {
        "ParamEyeLOpen": 0.35,
        "ParamEyeROpen": 0.35,
        "ParamEyeLSmile": 0,
        "ParamEyeRSmile": 0,
        "ParamMouthForm": -1.0,
        "ParamBrowLY": -0.8,
        "ParamBrowRY": -0.8,
        "ParamBrowLAngle": -0.6,
        "ParamBrowRAngle": -0.6,
        "ParamAngleY": -15,
        "ParamAngleX": -5,
        "ParamBodyAngleY": -4,
    },
    "焦虑": {
        "ParamEyeLOpen": 1,
        "ParamEyeROpen": 1,
        "ParamBrowLY": 0.8,
        "ParamBrowRY": 0.8,
        "ParamBrowLAngle": 0.7,
        "ParamBrowRAngle": 0.7,
        "ParamMouthForm": -0.3,
        "ParamBodyAngleX": 4,
        "ParamAngleX": 5,
    },
    "孤独": {
        "ParamEyeLOpen": 0.45,
        "ParamEyeROpen": 0.45,
        "ParamAngleZ": -12,
        "ParamAngleY": -10,
        "ParamBrowLY": -0.4,
        "ParamBrowRY": -0.4,
        "ParamMouthForm": -0.4,
        "ParamBodyAngleZ": -3,
    },
    "愤怒": {
        "ParamBrowLY": -1.0,
        "ParamBrowRY": -1.0,
        "ParamBrowLAngle": -1.0,
        "ParamBrowRAngle": -1.0,
        "ParamEyeLOpen": 1.0,
        "ParamEyeROpen": 1.0,
        "ParamMouthForm": -0.8,
        "ParamAngleY": 6,
        "ParamAngleX": -3,
        "ParamBodyAngleX": -3,
    },
    "疲惫": {
        "ParamEyeLOpen": 0.25,
        "ParamEyeROpen": 0.25,
        "ParamBrowLY": -0.5,
        "ParamBrowRY": -0.5,
        "ParamAngleY": -18,
        "ParamBreath": 0.8,
        "ParamMouthForm": -0.2,
        "ParamBodyAngleY": -3,
    },
    "困惑": {
        "ParamBrowLY": 0.8,
        "ParamBrowRY": -0.5,
        "ParamBrowLAngle": 0.7,
        "ParamBrowRAngle": -0.4,
        "ParamAngleZ": -15,
        "ParamEyeLOpen": 1.0,
        "ParamEyeROpen": 0.6,
        "ParamMouthForm": -0.3,
        "ParamBodyAngleZ": -2,
    },
    "平静": {
        "ParamBreath": 0.5,
        "ParamEyeLSmile": 0.15,
        "ParamEyeRSmile": 0.15,
        "ParamMouthForm": 0.1,
    },
}

# 高强度时可触发的姿势参数 {emotion: (param_id, threshold)}
# 姿势名称需要与实际的 Live2D 模型参数匹配，如果不确定可设为 None
EMOTION_POSES = {
    "开心": ("ParamPOSE2", 0.75),  # 打招呼
    "难过": ("ParamPOSE7", 0.6),   # 无奈摊手
    "焦虑": ("ParamPOSE4", 0.5),   # 思考/不安
    "孤独": ("ParamPOSE8", 0.4),   # 低落
    "愤怒": ("ParamPOSE1", 0.7),   # 抬手
    "疲惫": ("ParamPOSE5", 0.3),   # 无力
    "困惑": ("ParamPOSE4", 0.5),   # 思考
    "平静": (None, 0),             # 平静不需要特殊姿势
}


def get_live2d_params(emotion: str, intensity: float) -> dict:
    """
    根据情绪和强度计算 Live2D 参数。
    返回 {"params": {参数字典}, "pose": 姿势ID或None}
    """
    intensity = max(0.0, min(1.0, intensity))
    target = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["平静"])

    # 从中性基线向目标插值
    params = {}
    for key, neutral_val in NEUTRAL_PARAMS.items():
        target_val = target.get(key, neutral_val)
        params[key] = neutral_val + (target_val - neutral_val) * intensity

    # 检查是否触发姿势
    pose = None
    if emotion in EMOTION_POSES:
        pose_id, threshold = EMOTION_POSES[emotion]
        if intensity >= threshold:
            pose = pose_id

    return {"params": params, "pose": pose}


class Live2DSmoother:
    """Live2D 参数平滑器 (低通滤波)"""
    def __init__(self, alpha: float = 0.3):
        self.current_params = NEUTRAL_PARAMS.copy()
        self.alpha = alpha  # 平滑系数 (0.0-1.0)，越小越平滑但延迟越高

    def smooth(self, target_params: dict, alpha: float = None) -> dict:
        """平滑过渡到目标参数

        Args:
            target_params: 目标参数字典
            alpha: 平滑系数，如果为None则使用self.alpha
        """
        if alpha is None:
            alpha = self.alpha

        smoothed = {}
        # 遍历所有目标参数
        for key, target_val in target_params.items():
            # 获取当前值（如果没有则默认为目标值，避免第一帧跳变）
            current_val = self.current_params.get(key, target_val)

            # 低通滤波: y[i] = α * x[i] + (1-α) * y[i-1]
            new_val = alpha * target_val + (1 - alpha) * current_val

            smoothed[key] = new_val
            self.current_params[key] = new_val

        # 确保包含所有 key (以防 target_params 不完整)
        # 这里假设 target_params 通常是完整的 (from get_live2d_params)

        return smoothed

