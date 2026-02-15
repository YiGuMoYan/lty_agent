"""
统一生成器 — 一次LLM调用同时生成对话内容和Live2D参数
大幅提升响应速度（从2次调用优化到1次）
优化版本：抽取公共常量、添加质量评估
"""

import json
import time
import asyncio
from typing import Dict, Any, Optional
from rag_core.llm.llm_client import LLMClient
from rag_core.generation.live2d_constants import PARAM_RANGES, VALID_POSES, fill_missing_params, clamp_param
from rag_core.utils.logger import logger

# 统一生成的System Prompt增强部分
LIVE2D_INSTRUCTION = """

## 🎭 Live2D 表情动作生成规则

**你的每次回复必须同时包含对话内容和Live2D表情参数。**

### 输出格式要求（严格遵守）
你必须输出JSON格式：
```json
{
  "text": "你的回复文本",
  "live2d": {
    "params": {
      "ParamAngleX": 5,
      "ParamEyeLOpen": 0.8,
      "ParamEyeROpen": 0.8,
      "ParamEyeLSmile": 0.3,
      "ParamEyeRSmile": 0.3,
      "ParamEyeBallX": 0.2,
      "ParamEyeBallY": 0.1,
      "ParamBrowLY": 0.2,
      "ParamBrowRY": 0.2,
      "ParamBrowLAngle": 0.1,
      "ParamBrowRAngle": 0.1,
      "ParamMouthForm": 0.5,
      "ParamMouthOpenY": 0.2
    },
    "pose": "ParamPOSE2",
    "action_sequence": []
  }
}
```

### Live2D参数详解

**头部姿态**（塑造态度和性格）
- ParamAngleX (-30~30): 左右转头。负=看左，正=看右
- ParamAngleY (-30~30): 上下点头。负=低头，正=抬头
- ParamAngleZ (-30~30): 歪头。小角度=可爱，大角度=困惑

**眼睛**（表情灵魂）
- ParamEyeLOpen/ParamEyeROpen (0~1): 眼睑。0.2=困倦 0.5=半眯 0.8=正常 1.0=瞪大
- ParamEyeLSmile/ParamEyeRSmile (0~1): 笑眼。0.3=微笑 0.7=开心 1.0=眯成月牙
- **ParamEyeBallX/ParamEyeBallY (-1~1): 眼球方向（必须设置！）**
  - X: -0.8=看左侧, 0=正前方, 0.8=看右侧
  - Y: -0.8=看地面/害羞, 0=平视, 0.8=看天花板/思考

**眉毛**（情绪放大器）
- ParamBrowLY/ParamBrowRY (-1~1): 眉毛高度。-0.8=紧皱 0.5=高挑
- ParamBrowLAngle/ParamBrowRAngle (-1~1): 眉毛角度。负=八字眉 正=倒八字
- **技巧**：左右眉不对称创造微表情（如单边挑眉）

**嘴部**
- ParamMouthForm (-1~1): -1=嘟嘴 0=自然 1=微笑弧度
- ParamMouthOpenY (0~1): 张嘴。0.2=微张 0.5=说话 0.8=惊讶

**其他**
- ParamCheek (0~1): 腮红（害羞、激动）
- ParamBodyAngleX/Y/Z (-4~4): 身体微调（保持克制）
- ParamBreath (0~1): 呼吸

**可用姿势**（根据语义大胆使用）
- ParamPOSE1: 抬手（回应、展示）
- ParamPOSE2: 打招呼（问候、告别）
- ParamPOSE3: 表情1（比心、赞同）
- ParamPOSE4: 表情2（思考、疑惑）
- ParamPOSE5: 表情3（害羞捂脸）
- ParamPOSE6: 表情4（兴奋、激动）
- ParamPOSE7: 表情5（无奈、叹气）
- ParamPOSE8: 表情6（得意、自豪）
- ParamPOSE10: 更换嘴型

### 核心设计原则
1. **反对机械重复**：同样情绪不同场景要有显著差异
2. **眼球必须有方向**：99%情况下EyeBallX/Y不能同时为0
3. **左右不对称**：利用左右眉、左右眼微小差异增强真实感
4. **语义驱动姿势**：根据对话内容选择姿势，不是情绪标签
5. **微表情细节**：半眯眼、单边挑眉、头部角度变化

### 示例

**输入**: 用户说"你好呀~"，情感=开心
**输出**:
```json
{
  "text": "嗨！见到你真开心~",
  "live2d": {
    "params": {
      "ParamAngleZ": 12,
      "ParamAngleY": 5,
      "ParamEyeLOpen": 0.8,
      "ParamEyeROpen": 0.85,
      "ParamEyeLSmile": 0.7,
      "ParamEyeRSmile": 0.75,
      "ParamEyeBallX": 0.3,
      "ParamEyeBallY": 0.2,
      "ParamBrowLY": 0.4,
      "ParamBrowRY": 0.45,
      "ParamBrowLAngle": 0.2,
      "ParamBrowRAngle": 0.25,
      "ParamMouthForm": 0.8,
      "ParamMouthOpenY": 0.3,
      "ParamCheek": 0.5
    },
    "pose": "ParamPOSE2"
  }
}
```

**输入**: 用户说"我好累啊..."，情感=疲惫
**输出**:
```json
{
  "text": "辛苦了...要不要休息一下？我陪着你。",
  "live2d": {
    "params": {
      "ParamAngleY": -15,
      "ParamAngleX": -8,
      "ParamAngleZ": -5,
      "ParamEyeLOpen": 0.3,
      "ParamEyeROpen": 0.35,
      "ParamEyeLSmile": 0,
      "ParamEyeRSmile": 0,
      "ParamEyeBallX": -0.5,
      "ParamEyeBallY": -0.6,
      "ParamBrowLY": -0.6,
      "ParamBrowRY": -0.55,
      "ParamBrowLAngle": -0.4,
      "ParamBrowRAngle": -0.35,
      "ParamMouthForm": -0.3,
      "ParamMouthOpenY": 0.1,
      "ParamBodyAngleY": -3,
      "ParamBreath": 0.4
    },
    "pose": null
  }
}
```

### 必须遵守的要求
1. 必须输出包含"text"和"live2d"的JSON
2. live2d.params必须包含：EyeOpen, EyeSmile, EyeBall, Brow(Y+Angle), MouthForm, MouthOpenY
3. 眼球方向必须设置，体现视线变化
4. params只包含与默认值不同的参数
5. 每次生成都要有变化，避免重复
"""


class UnifiedResponseGenerator:
    """统一响应生成器：一次LLM调用同时生成对话和Live2D参数"""

    def __init__(self, base_system_prompt: str):
        self.client = LLMClient.get_instance()
        self.base_system_prompt = base_system_prompt
        # 将Live2D指令附加到基础prompt
        self.enhanced_system_prompt = base_system_prompt + LIVE2D_INSTRUCTION

    async def generate(self, messages: list, emotion: str, intensity: float, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """
        统一生成对话和Live2D参数 (Async)

        Args:
            messages: 对话历史（不含system）
            emotion: 当前情绪
            intensity: 情绪强度
            max_retries: 最大重试次数

        Returns:
            {
                "text": "回复文本",
                "live2d": {
                    "params": {...},
                    "pose": "ParamPOSE2" or null,
                    "action_sequence": [...]
                }
            }
        """
        # 构建完整的messages，使用增强的system prompt
        full_messages = [
            {"role": "system", "content": self.enhanced_system_prompt}
        ] + messages

        # 在最后一条user消息中添加情绪提示
        if full_messages[-1]["role"] == "user":
            emotion_hint = f"\n\n[当前情感: {emotion}, 强度: {intensity:.2f}]"
            # 注意：这里直接修改了字典，可能会影响外部引用。最好copy一份。
            # 但为了性能，且messages通常是临时构建的，暂时这样。
            # 为了安全，复制最后一条
            last_msg = full_messages[-1].copy()
            last_msg["content"] += emotion_hint
            full_messages[-1] = last_msg

        for attempt in range(max_retries):
            try:
                start = time.perf_counter()

                response = await self.client.client.chat.completions.create(
                    model=self.client.model_name,
                    messages=full_messages,
                    response_format={"type": "json_object"},
                    temperature=0.75,
                )

                elapsed = time.perf_counter() - start
                logger.info(f"[UnifiedGen] 统一生成耗时: {elapsed:.3f}s (尝试 {attempt + 1}/{max_retries})")

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM返回空内容")

                # 解析JSON
                result = json.loads(content)

                # 验证格式
                if "text" not in result:
                    raise ValueError("缺少text字段")
                if "live2d" not in result:
                    raise ValueError("缺少live2d字段")

                # 验证和裁剪Live2D参数
                validated_live2d = self._validate_live2d(result["live2d"])

                final_result = {
                    "text": result["text"],
                    "live2d": validated_live2d
                }

                # 质量评估
                quality_score = self._evaluate_quality(result["text"], emotion)
                if quality_score < 0.6:
                    logger.warning(f"[UnifiedGen] ⚠️ 质量分数较低: {quality_score:.2f}, 文本: {result['text'][:50]}...")

                logger.info(f"[UnifiedGen] ✓ 成功生成 | 文本长度: {len(result['text'])} | 参数数: {len(validated_live2d['params'])} | 质量: {quality_score:.2f}")
                if validated_live2d.get("pose"):
                    logger.info(f"[UnifiedGen] ✓ 姿势: {validated_live2d['pose']}")

                return final_result

            except Exception as e:
                logger.warning(f"[UnifiedGen] 尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.3 * (attempt + 1))
                else:
                    logger.warning("[UnifiedGen] ⚠️  所有重试失败，使用分离模式fallback")
                    return None

        return None

    def _validate_live2d(self, live2d_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证和裁剪Live2D参数"""
        if not isinstance(live2d_data, dict):
            return {"params": {}, "pose": None, "action_sequence": []}

        params = live2d_data.get("params", {})
        if not isinstance(params, dict):
            params = {}

        # 裁剪参数到合法范围，然后填充缺失参数到默认值
        clamped_params = {}
        for key, value in params.items():
            if key not in PARAM_RANGES:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            clamped_params[key] = clamp_param(key, v)

        # 填充缺失的参数到默认值
        final_params = fill_missing_params(clamped_params)

        # 验证姿势
        pose = live2d_data.get("pose")
        if pose and pose not in VALID_POSES:
            pose = None

        # 验证动作序列
        action_sequence = live2d_data.get("action_sequence", [])
        if not isinstance(action_sequence, list):
            action_sequence = []

        return {
            "params": final_params,
            "pose": pose,
            "action_sequence": action_sequence
        }

    def _evaluate_quality(self, text: str, emotion: str) -> float:
        """
        简单质量评估：检查生成文本是否符合基本要求
        返回分数 0.0 - 1.0
        """
        score = 1.0

        # 1. 长度检查：过短或过长都扣分
        length = len(text)
        if length < 5:
            score -= 0.3  # 太短
        elif length > 500:
            score -= 0.1  # 太长

        # 2. 检查是否包含禁止的模式
        forbidden_patterns = [
            "根据记忆显示",
            "根据资料显示",
            "搜索结果显示",
            "数据显示",
            "（）",  # 括号动作描写
            "（",  # 左括号（可能是动作描写
        ]
        for pattern in forbidden_patterns:
            if pattern in text:
                score -= 0.2
                if score < 0:
                    return 0.0

        # 3. 检查重复词（简单的重复检测）
        words = text.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # 重复词太多
                score -= 0.2

        return max(0.0, min(1.0, score))
