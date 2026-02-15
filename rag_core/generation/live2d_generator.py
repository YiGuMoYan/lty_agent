"""
Live2D 参数生成器 — 通过 LLM 根据回复文本和情感状态动态生成 Live2D 模型参数
支持动作多样化、微表情、动作序列等高级特性
优化版本：使用公共常量、改进多样性策略
"""

import json
import time
import random
from typing import Optional, Dict, Any, List
from rag_core.llm.llm_client import LLMClient
from rag_core.generation.live2d_constants import PARAM_RANGES, VALID_POSES, fill_missing_params, clamp_param

SYSTEM_PROMPT = """你是 Live2D 虚拟形象的**高级表情动作导演**。
你的任务是根据角色的回复文本和情感状态，创造**生动、自然、多变**的表情和动作组合。

## 🎭 核心设计理念
1. **反对机械重复**：同样的情绪在不同语境下必须有显著差异
2. **微表情丰富度**：细节决定真实感（眼球方向、眉毛微调、头部角度）
3. **动作叙事性**：姿势和表情必须契合对话语义，而非简单映射
4. **多样性优先**：宁可冒险尝试新组合，也不要安全但平庸的重复

## 📐 参数系统详解

### 头部姿态（塑造性格和态度）
- ParamAngleX (-30~30): 左右转头。负=看左，正=看右
- ParamAngleY (-30~30): 上下点头。负=低头/失落，正=抬头/自信
- ParamAngleZ (-30~30): 歪头。小角度=可爱，大角度=困惑/质疑

### 眼睛（表情的灵魂）
- ParamEyeLOpen / ParamEyeROpen (0~1): 眼睑高度
  * 0.0-0.2: 闭眼/困倦  * 0.3-0.5: 半眯/慵懒  * 0.6-0.9: 正常
  * 1.0: 瞪大/惊讶
- ParamEyeLSmile / ParamEyeRSmile (0~1): 笑眼程度
  * 0.3-0.5: 温柔微笑  * 0.6-0.8: 开心笑眼  * 0.9-1.0: 眯成月牙
- **ParamEyeBallX / ParamEyeBallY (-1~1): 眼球方向（必须设置！）**
  * X: -0.8=看左侧, 0=看正前方, 0.8=看右侧
  * Y: -0.8=看地面/害羞, 0=平视, 0.8=看天花板/思考
  * **重要**：眼球方向传达潜台词（回避、期待、思考、警惕）

### 眉毛（情绪的放大器）
- ParamBrowLY / ParamBrowRY (-1~1): 眉毛高度
  * -0.8~-1.0: 紧皱/愤怒  * 0.5-0.8: 高挑/惊讶  * 0: 自然
- ParamBrowLAngle / ParamBrowRAngle (-1~1): 眉毛角度
  * 负值: 八字眉(外低内高)/委屈  * 正值: 倒八字/疑惑
  * **技巧**：左右眉不对称能创造"挑眉"等微表情

### 嘴部
- ParamMouthForm (-1~1): 嘴型基调
  * -1.0~-0.5: 嘟嘴/不满  * 0: 自然  * 0.5-1.0: 微笑弧度
- ParamMouthOpenY (0~1): 张嘴程度
  * 0.2-0.3: 微张  * 0.4-0.6: 说话  * 0.7-1.0: 惊讶/大笑

### 其他细节
- ParamCheek (0~1): 腮红（害羞、激动、醉酒）
- ParamBodyAngleX/Y/Z (-4~4): 身体微调（配合头部，不可过度）
- ParamBreath (0~1): 呼吸（平静、疲惫、喘息）

## 🎬 姿势系统（动作多样化的关键）
可用姿势（请根据语义**大胆使用**）：
- ParamPOSE1: 抬手（回应、打断、展示）
- ParamPOSE2: 打招呼（问候、告别、呼唤）
- ParamPOSE3: 表情1（比心、赞同）
- ParamPOSE4: 表情2（思考手托腮、疑惑）
- ParamPOSE5: 表情3（害羞捂脸、不好意思）
- ParamPOSE6: 表情4（兴奋、激动）
- ParamPOSE7: 表情5（无奈摊手、叹气）
- ParamPOSE8: 表情6（得意、自豪）
- ParamPOSE10: 更换嘴型（特殊表情）

### 姿势使用原则
1. 不要保守！语义有30%匹配就可以尝试使用
2. 同样的情绪可以用不同姿势（开心可以是POSE2/POSE3/POSE6/POSE8）
3. 优先使用动作让角色"动起来"，而非只靠面部参数
4. 可以组合使用：pose + 对应的面部微表情强化效果

## 🎨 多样化创作策略

### 策略1：同情绪不同表现
**开心**可以是：
- 眼睛瞪大+看向上方(期待型开心) → POSE6
- 眼睛眯起+腮红(害羞型开心) → POSE5
- 眼球看向侧面+歪头(调皮型开心) → POSE8
- 眉毛高挑+微张嘴(惊喜型开心) → POSE2

### 策略2：语义驱动姿势
不要机械对应！根据**对话内容**选择：
- "你说的对" → POSE1(认同手势)
- "哎呀~" → POSE5(害羞捂脸)
- "让我想想..." → POSE4(思考)
- "太棒了！" → POSE6(兴奋)
- "没办法呢" → POSE7(无奈)

### 策略3：随机微调
即使相同情绪强度，也要加入变化：
- 头部角度：±5度随机偏移
- 眼球方向：避免总是(0,0)，添加视线变化
- 眉毛：左右微调0.1-0.2的差值

## 📋 输出格式
严格JSON:
{
  "params": {参数字典},
  "pose": "ParamPOSE2" 或 null,
  "action_sequence": [可选的动作序列]
}

### action_sequence（可选）
当对话需要**连续动作**时使用：
[
  {"delay_ms": 0, "pose": "ParamPOSE4", "duration_ms": 800},
  {"delay_ms": 1000, "params": {"ParamAngleZ": -15}, "duration_ms": 600},
  {"delay_ms": 1800, "pose": null}
]

## ✅ 质量检查清单
生成前自查：
- [ ] 是否设置了EyeBallX/Y（不能都是0！）
- [ ] 左右眉毛是否有细微差异
- [ ] 姿势是否契合对话语义（不是情绪类别）
- [ ] 与之前的回复相比，参数组合是否有显著差异
- [ ] 头部角度是否有变化（不要总是中正）
- [ ] 是否利用了微表情细节（半眯眼、单边挑眉等）

## 🌟 示例（注意多样性）

输入: 情感=开心, 强度=0.85, 回复="太好了！这正是我想要的！"
输出: {"params": {"ParamAngleZ": 12, "ParamAngleY": 8, "ParamEyeLOpen": 1.0, "ParamEyeROpen": 0.95, "ParamEyeLSmile": 0.6, "ParamEyeRSmile": 0.7, "ParamEyeBallX": -0.3, "ParamEyeBallY": 0.4, "ParamMouthForm": 1.0, "ParamMouthOpenY": 0.65, "ParamBrowLY": 0.5, "ParamBrowRY": 0.6, "ParamBrowLAngle": 0.2, "ParamBrowRAngle": 0.25, "ParamCheek": 0.6, "ParamBodyAngleZ": 3}, "pose": "ParamPOSE6"}

输入: 情感=开心, 强度=0.75, 回复="嘿嘿，我就知道~"（同样是开心但表现不同）
输出: {"params": {"ParamAngleZ": -18, "ParamAngleY": -6, "ParamEyeLOpen": 0.4, "ParamEyeROpen": 0.45, "ParamEyeLSmile": 0.95, "ParamEyeRSmile": 0.9, "ParamEyeBallX": -0.7, "ParamEyeBallY": -0.3, "ParamMouthForm": 0.7, "ParamMouthOpenY": 0.2, "ParamBrowLY": 0.15, "ParamBrowRY": 0.25, "ParamBrowLAngle": 0.1, "ParamBrowRAngle": 0.3, "ParamCheek": 0.85, "ParamBodyAngleY": -2}, "pose": "ParamPOSE8"}

输入: 情感=困惑, 强度=0.70, 回复="诶？这是怎么回事？"
输出: {"params": {"ParamAngleZ": -20, "ParamAngleX": -8, "ParamEyeLOpen": 0.95, "ParamEyeROpen": 0.5, "ParamEyeLSmile": 0, "ParamEyeRSmile": 0, "ParamEyeBallX": 0.6, "ParamEyeBallY": 0.4, "ParamBrowLY": 0.85, "ParamBrowRY": -0.3, "ParamBrowLAngle": 0.7, "ParamBrowRAngle": -0.5, "ParamMouthForm": -0.25, "ParamMouthOpenY": 0.5, "ParamBodyAngleZ": -3}, "pose": "ParamPOSE4"}

输入: 情感=难过, 强度=0.60, 回复="算了...没关系的。"
输出: {"params": {"ParamAngleY": -22, "ParamAngleX": -12, "ParamAngleZ": -8, "ParamEyeLOpen": 0.25, "ParamEyeROpen": 0.2, "ParamEyeLSmile": 0, "ParamEyeRSmile": 0, "ParamEyeBallY": -0.85, "ParamEyeBallX": -0.6, "ParamMouthForm": -0.7, "ParamMouthOpenY": 0.05, "ParamBrowLY": -0.9, "ParamBrowRY": -0.85, "ParamBrowLAngle": -0.65, "ParamBrowRAngle": -0.7, "ParamBodyAngleY": -4, "ParamBreath": 0.3}, "pose": "ParamPOSE7"}

输入: 情感=兴奋, 强度=0.90, 回复="快看快看！那边有好东西！"
输出: {"params": {"ParamAngleY": 10, "ParamAngleX": 18, "ParamAngleZ": 15, "ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0, "ParamEyeLSmile": 0.3, "ParamEyeRSmile": 0.35, "ParamEyeBallX": 0.8, "ParamEyeBallY": 0.2, "ParamMouthForm": 0.9, "ParamMouthOpenY": 0.7, "ParamBrowLY": 0.7, "ParamBrowRY": 0.75, "ParamBrowLAngle": 0.3, "ParamBrowRAngle": 0.35, "ParamCheek": 0.5, "ParamBodyAngleX": 3, "ParamBreath": 0.8}, "pose": "ParamPOSE1", "action_sequence": [{"delay_ms": 0, "pose": "ParamPOSE1", "duration_ms": 600}, {"delay_ms": 800, "params": {"ParamAngleX": 22}, "duration_ms": 400}]}

## ⚠️ 最高优先级要求
1. **绝对禁止重复**：连续两次相同情绪绝不能产生相似参数组合
2. **眼球必须有方向**：99%的情况下 EyeBallX/Y 不能同时为0
3. **大胆使用姿势**：只要语义有关联就用，不要保守
4. **左右不对称**：利用左右眉、左右眼的微小差异创造真实感
5. **结合语义**：不要只看emotion标签，更要分析reply_text的具体含义"""


class Live2DParamGenerator:
    def __init__(self):
        self.client = LLMClient.get_instance()
        self.history: List[Dict[str, Any]] = []  # 记录最近的生成历史，避免重复

    def generate(self, reply_text: str, emotion: str, intensity: float, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        通过 LLM 生成 Live2D 参数（带重试和 fallback）。

        Args:
            reply_text: 角色的回复文本
            emotion: 情感类别（开心/难过/焦虑等）
            intensity: 情感强度 0.0-1.0
            max_retries: 最大重试次数

        Returns:
            {"params": {...}, "pose": ..., "action_sequence": [...]} 或静态映射
        """
        # 添加随机种子和历史上下文，增强多样性
        diversity_hint = self._generate_diversity_hint()

        user_prompt = (
            f"角色回复: \"{reply_text}\"\n"
            f"情感: {emotion}\n"
            f"强度: {intensity:.2f}\n\n"
            f"创作要求：\n"
            f"1. 必须包含全部面部参数: EyeLOpen, EyeROpen, EyeLSmile, EyeRSmile, EyeBallX, EyeBallY, BrowLY, BrowRY, BrowLAngle, BrowRAngle, MouthForm, MouthOpenY\n"
            f"2. 眼球方向(EyeBallX/Y)必须非零，体现视线变化\n"
            f"3. 根据对话语义选择合适的姿势(pose)\n"
            f"4. 避免重复和机械感\n"
            f"{diversity_hint}"
        )

        for attempt in range(max_retries):
            try:
                start = time.perf_counter()
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

                # 提高 temperature 增强多样性
                temperature = 0.75 + random.uniform(0, 0.15)

                response = self.client.client.chat.completions.create(
                    model=self.client.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )

                content = response.choices[0].message.content
                elapsed = time.perf_counter() - start
                print(f"[Live2DGen] LLM 生成耗时: {elapsed:.3f}s (尝试 {attempt + 1}/{max_retries}, temp={temperature:.2f})")

                if content is None:
                    raise ValueError("LLM 返回空内容")

                print(f"[Live2DGen] LLM 原始输出: {content[:200]}...")
                result = json.loads(content)
                validated = self._validate_and_clamp(result)

                if validated:
                    # 添加微表情随机扰动
                    validated = self._add_micro_variations(validated)

                    # 记录到历史
                    self._add_to_history(validated, emotion, intensity)

                    non_zero = {k: round(v, 3) for k, v in validated["params"].items() if v != 0}
                    print(f"[Live2DGen] ✓ 情感={emotion} 强度={intensity:.2f} → {len(non_zero)}个参数")
                    if validated.get("pose"):
                        print(f"[Live2DGen] ✓ 姿势: {validated['pose']}")
                    if validated.get("action_sequence"):
                        print(f"[Live2DGen] ✓ 动作序列: {len(validated['action_sequence'])}步")

                    return validated
                else:
                    raise ValueError("参数验证失败")

            except Exception as e:
                print(f"[Live2DGen] 尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.3 * (attempt + 1))
                else:
                    print("[Live2DGen] ⚠️  所有重试失败，回退到静态映射")
                    return self._fallback_static(emotion, intensity, reply_text)

        return self._fallback_static(emotion, intensity, reply_text)

    def _generate_diversity_hint(self) -> str:
        """
        根据历史生成多样性提示（优化版本）
        不再强制避免相同组合，而是添加适度随机扰动
        """
        if len(self.history) < 2:
            return ""

        recent = self.history[-2:]
        hints = ["\n多样性创作提示:"]

        # 添加适度随机建议，而不是强制避免
        hints.append("- 在保持语义一致的前提下，可以尝试新的组合")
        hints.append("- 加入微小的随机扰动让表情更生动")

        # 提供历史参考，但只是作为"不要完全相同"的提示
        recent_poses = [h.get("pose") for h in recent if h.get("pose")]
        if recent_poses:
            hints.append(f"- 参考：最近用了 {recent_poses[-1]}，这次可以尝试不同的风格")

        return "\n".join(hints)

    def _add_micro_variations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """添加微表情随机扰动，增强自然感"""
        params = result.get("params", {})

        # 对部分参数添加 ±5% 的随机扰动
        variation_keys = ["ParamAngleZ", "ParamEyeBallX", "ParamEyeBallY", "ParamBrowLAngle", "ParamBrowRAngle"]
        for key in variation_keys:
            if key in params:
                original = params[key]
                variation = random.uniform(-0.05, 0.05) * abs(original) if original != 0 else random.uniform(-0.1, 0.1)
                params[key] = original + variation

        # 确保左右眉有细微差异
        if "ParamBrowLY" in params and "ParamBrowRY" in params:
            if abs(params["ParamBrowLY"] - params["ParamBrowRY"]) < 0.05:
                params["ParamBrowRY"] += random.uniform(-0.1, 0.1)

        result["params"] = params
        return result

    def _add_to_history(self, result: Dict[str, Any], emotion: str, intensity: float):
        """记录生成历史（保留最近10条）"""
        self.history.append({
            "params": result.get("params", {}),
            "pose": result.get("pose"),
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": time.time()
        })
        if len(self.history) > 10:
            self.history.pop(0)

    def _fallback_static(self, emotion: str, intensity: float, reply_text: str) -> Dict[str, Any]:
        """静态映射作为 fallback（增强版）"""
        from emotion_live2d_map import get_live2d_params

        print(f"[Live2DGen] 使用静态映射: {emotion} @ {intensity:.2f}")
        static_result = get_live2d_params(emotion, intensity)

        # 为静态结果添加随机性
        params = static_result.get("params", {})

        # 添加随机眼球方向（避免总是看正前方）
        if "ParamEyeBallX" not in params or params["ParamEyeBallX"] == 0:
            params["ParamEyeBallX"] = random.uniform(-0.4, 0.4)
        if "ParamEyeBallY" not in params or params["ParamEyeBallY"] == 0:
            params["ParamEyeBallY"] = random.uniform(-0.3, 0.3)

        # 添加随机头部角度微调
        params["ParamAngleZ"] = params.get("ParamAngleZ", 0) + random.uniform(-5, 5)

        # 根据语义选择姿势
        pose = self._infer_pose_from_text(reply_text, emotion)

        return {
            "params": params,
            "pose": pose or static_result.get("pose"),
            "action_sequence": []
        }

    def _infer_pose_from_text(self, text: str, emotion: str) -> Optional[str]:
        """从文本推断合适的姿势"""
        text_lower = text.lower()

        # 关键词匹配
        if any(kw in text_lower for kw in ["你好", "嗨", "hi", "hello", "再见", "拜拜"]):
            return "ParamPOSE2"
        if any(kw in text_lower for kw in ["嘿嘿", "害羞", "不好意思", "///", "脸红"]):
            return "ParamPOSE5"
        if any(kw in text_lower for kw in ["想想", "思考", "让我", "考虑", "嗯..."]):
            return "ParamPOSE4"
        if any(kw in text_lower for kw in ["太棒", "好耶", "太好了", "开心", "耶"]):
            return "ParamPOSE6"
        if any(kw in text_lower for kw in ["算了", "唉", "没办法", "无奈"]):
            return "ParamPOSE7"
        if any(kw in text_lower for kw in ["当然", "必须", "就是", "没错", "得意"]):
            return "ParamPOSE8"

        # 根据情绪随机选择
        emotion_pose_map = {
            "开心": random.choice(["ParamPOSE2", "ParamPOSE3", "ParamPOSE6", None]),
            "难过": random.choice(["ParamPOSE7", None]),
            "困惑": random.choice(["ParamPOSE4", None]),
            "焦虑": random.choice(["ParamPOSE4", None]),
        }
        return emotion_pose_map.get(emotion)

    def _validate_and_clamp(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """校验并裁剪参数到合法范围"""
        if not isinstance(result, dict) or "params" not in result:
            print("[Live2DGen] 无效的输出格式，缺少 params 字段")
            return None

        raw_params = result["params"]
        if not isinstance(raw_params, dict):
            print("[Live2DGen] params 不是字典")
            return None

        clamped_params = {}
        for key, value in raw_params.items():
            if key not in PARAM_RANGES:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            lo, hi = PARAM_RANGES[key]
            clamped_params[key] = max(lo, min(hi, v))

        pose = result.get("pose", None)
        if pose is not None and pose not in VALID_POSES:
            pose = None

        # 验证 action_sequence
        action_sequence = result.get("action_sequence", [])
        if not isinstance(action_sequence, list):
            action_sequence = []

        return {
            "params": clamped_params,
            "pose": pose,
            "action_sequence": action_sequence
        }
