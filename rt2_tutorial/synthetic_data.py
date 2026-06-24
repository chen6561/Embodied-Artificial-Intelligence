from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from rt2_tokenizer import RT2Tokenizer


@dataclass
class SyntheticSceneConfig:
    """
    合成场景配置。

    这里的场景非常简单：黑色背景上放 3 个彩色方块。
    这样我们可以很轻松地构造：
    - VQA 风格问题：例如“最左边是什么颜色”
    - 机器人控制问题：例如“去抓红色目标”
    """

    image_size: int = 32
    num_frames: int = 3
    square_size: int = 5


@dataclass
class SceneObject:
    """
    场景里的一个彩色方块目标。
    """

    color_name: str
    color_value: tuple[float, float, float]
    center_x: int
    center_y: int


class SyntheticSceneFactory:
    """
    负责生成简单的几何场景和对应的监督信号。

    整个教学工程里最关键的目标不是造一个“真实机器人数据集”，
    而是造一个足够可控的小世界，让 RT-2 的统一建模思路可以跑通。
    """

    def __init__(self, config: SyntheticSceneConfig) -> None:
        self.config = config
        self.colors = [
            ("red", (1.0, 0.2, 0.2)),
            ("green", (0.2, 1.0, 0.2)),
            ("blue", (0.2, 0.4, 1.0)),
        ]

    def _sample_centers(self, index: int) -> list[tuple[int, int]]:
        """
        确定性地生成 3 个不会严重重叠的目标中心。

        之前这里用 while 随机试探去找不重叠的位置：
        1. 理论上能工作；
        2. 但在调试时可能偶尔花很久，表现得像“程序卡住”。

        现在改成纯确定性布局：
        - 不再依赖 rejection sampling
        - 不会出现数据集构造阶段的死等
        - 同一个 index 总是得到同一个场景
        """

        margin = self.config.square_size + 2
        size = self.config.image_size - margin - 1

        # 我们先在图像的四个角附近放置三个“基准点”。
        # 再根据 index 做一个小幅平移，这样既稳定又有变化。
        anchors = [
            (margin + 2, margin + 2),
            (size - 6, margin + 3),
            (margin + 3, size - 6),
        ]
        offsets = [
            (index % 3) - 1,
            ((index // 3) % 3) - 1,
            ((index // 9) % 3) - 1,
        ]

        centers: list[tuple[int, int]] = []
        for (base_x, base_y), offset in zip(anchors, offsets):
            dx = int(offset)
            dy = int(((index // 27) % 3) - 1)
            x = max(margin, min(size, base_x + dx))
            y = max(margin, min(size, base_y + dy))
            centers.append((x, y))
        return centers

    def build_scene(self, index: int) -> tuple[Tensor, list[SceneObject]]:
        """
        根据 index 生成一个可复现的合成场景。

        返回：
          images: [time, 3, H, W]
          objects: 3 个带颜色和坐标信息的目标
        """

        generator = torch.Generator().manual_seed(index)
        images = torch.zeros(
            self.config.num_frames,
            3,
            self.config.image_size,
            self.config.image_size,
        )

        centers = self._sample_centers(index)
        objects: list[SceneObject] = []
        for (color_name, color_value), (center_x, center_y) in zip(self.colors, centers):
            objects.append(
                SceneObject(
                    color_name=color_name,
                    color_value=color_value,
                    center_x=center_x,
                    center_y=center_y,
                )
            )

        # 给每一帧绘制同一场景，并加入很轻微的随机抖动。
        # 这样它看起来更像一个短历史，而不是完全重复的单帧复制。
        for frame_idx in range(self.config.num_frames):
            frame = torch.zeros(3, self.config.image_size, self.config.image_size)
            for obj in objects:
                jitter_x = int(torch.randint(-1, 2, (1,), generator=generator).item())
                jitter_y = int(torch.randint(-1, 2, (1,), generator=generator).item())
                self._draw_square(
                    frame,
                    center_x=max(0, min(self.config.image_size - 1, obj.center_x + jitter_x)),
                    center_y=max(0, min(self.config.image_size - 1, obj.center_y + jitter_y)),
                    color_value=obj.color_value,
                )

            # 加一点很弱的噪声，让视觉输入不至于过分僵硬。
            noise = torch.randn_like(frame) * 0.01
            images[frame_idx] = (frame + noise).clamp(0.0, 1.0)

        return images, objects

    def _draw_square(
        self,
        canvas: Tensor,
        center_x: int,
        center_y: int,
        color_value: tuple[float, float, float],
    ) -> None:
        """
        在画布上画一个小方块。
        """

        half = self.config.square_size // 2
        x0 = max(0, center_x - half)
        x1 = min(self.config.image_size, center_x + half + 1)
        y0 = max(0, center_y - half)
        y1 = min(self.config.image_size, center_y + half + 1)
        for channel, value in enumerate(color_value):
            canvas[channel, y0:y1, x0:x1] = value


class MixedRT2Dataset(Dataset):
    """
    一个混合数据集：
    - 一部分样本是“视觉问答”任务
    - 一部分样本是“机器人动作生成”任务

    这对应 RT-2 的 co-fine-tuning 核心思路：
    用统一模型同时学习 web-style VQA 数据和机器人控制数据。
    """

    def __init__(
        self,
        tokenizer: RT2Tokenizer,
        num_samples: int = 512,
        use_chain_of_thought: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.use_chain_of_thought = use_chain_of_thought
        self.scene_factory = SyntheticSceneFactory(SyntheticSceneConfig())

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        images, objects = self.scene_factory.build_scene(index)

        # 偶数样本当作 VQA 任务，奇数样本当作机器人任务。
        if index % 2 == 0:
            prompt_text, answer_text = self._build_vqa_sample(objects, index)
            prompt_ids = self.tokenizer.encode_text(prompt_text, add_eos=False)
            target_ids = self.tokenizer.encode_text(answer_text, add_eos=True)
            task_type = "vqa"
        else:
            prompt_text, action_values, plan_text = self._build_robot_sample(objects, index)
            prompt_ids = self.tokenizer.encode_text(prompt_text, add_eos=False)
            target_ids = self.tokenizer.build_robot_target(action_values, plan_text)
            task_type = "robot"

        return {
            "images": images,
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "task_type": task_type,
            "prompt_text": prompt_text,
        }

    def _build_vqa_sample(self, objects: list[SceneObject], index: int) -> tuple[str, str]:
        """
        构造一个简化的视觉问答任务。
        """

        if index % 4 == 0:
            leftmost = min(objects, key=lambda obj: obj.center_x)
            prompt = "task question answer what color is the leftmost object"
            answer = leftmost.color_name
        elif index % 4 == 2:
            rightmost = max(objects, key=lambda obj: obj.center_x)
            prompt = "task question answer what color is the rightmost object"
            answer = rightmost.color_name
        else:
            red_obj = next(obj for obj in objects if obj.color_name == "red")
            blue_obj = next(obj for obj in objects if obj.color_name == "blue")
            prompt = "task question answer is red left of blue"
            answer = "yes" if red_obj.center_x < blue_obj.center_x else "no"

        return prompt, answer

    def _build_robot_sample(
        self,
        objects: list[SceneObject],
        index: int,
    ) -> tuple[str, Tensor, str | None]:
        """
        构造一个简化的机器人控制任务。

        我们把“去抓某个颜色的目标”编码成一个 8 维连续动作：
          [x, y, z, roll, pitch, yaw, gripper, terminate]
        """

        color_cycle = ["red", "green", "blue"]
        target_color = color_cycle[(index // 2) % len(color_cycle)]
        target = next(obj for obj in objects if obj.color_name == target_color)

        prompt = f"task action pick the {target_color} object"
        action = self._build_action_from_object(target)

        # 可选的规划文本，对应 RT-2 论文里的 Action Chain-of-Thought 风格扩展。
        if self.use_chain_of_thought:
            plan_text = f"plan find {target_color} then grasp"
        else:
            plan_text = None

        return prompt, action, plan_text

    def _build_action_from_object(self, target: SceneObject) -> Tensor:
        """
        根据目标方块的位置构造一个教学版动作向量。

        这里的动作不追求真实机械臂动力学，只强调“图像位置 -> 动作 token”的映射。
        """

        size = float(self.scene_factory.config.image_size - 1)
        x_norm = target.center_x / size * 2.0 - 1.0
        y_norm = target.center_y / size * 2.0 - 1.0

        # 教学版里我们把 8 维动作解释为：
        # x / y / z / roll / pitch / yaw / gripper / terminate
        action = torch.tensor(
            [
                x_norm,
                y_norm,
                0.25,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ],
            dtype=torch.float32,
        )
        return action.clamp(-1.0, 1.0)


def build_language_model_batch(
    batch: list[dict[str, Tensor | str]],
    tokenizer: RT2Tokenizer,
) -> dict[str, Tensor | list[str]]:
    """
    把混合任务样本整理成统一的“语言模型训练批次”。

    关键点：
    - 图像序列单独作为视觉前缀输入
    - 文本/动作目标统一拼到 token 序列里
    - loss 只对 target 部分计算，prompt 部分全部忽略
    """

    images = torch.stack([sample["images"] for sample in batch])  # type: ignore[arg-type]
    prompt_texts = [sample["prompt_text"] for sample in batch]  # type: ignore[list-item]
    task_types = [sample["task_type"] for sample in batch]  # type: ignore[list-item]

    input_id_tensors: list[Tensor] = []
    label_tensors: list[Tensor] = []

    for sample in batch:
        prompt_ids = sample["prompt_ids"]  # type: ignore[assignment]
        target_ids = sample["target_ids"]  # type: ignore[assignment]

        # 统一构造成标准自回归语言模型格式：
        # full_sequence = [BOS] + prompt + target
        # input_ids     = full_sequence[:-1]
        # labels        = full_sequence[1:]
        full_sequence = torch.cat(
            [
                torch.tensor([tokenizer.bos_token_id], dtype=torch.long),
                prompt_ids,
                target_ids,
            ]
        )
        input_ids = full_sequence[:-1]
        labels = full_sequence[1:].clone()

        # prompt 部分不计算 loss，只在 target 部分监督。
        prompt_length = len(prompt_ids)
        labels[:prompt_length] = -100

        input_id_tensors.append(input_ids)
        label_tensors.append(labels)

    padded_input_ids = pad_sequence(
        input_id_tensors,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    padded_labels = pad_sequence(
        label_tensors,
        batch_first=True,
        padding_value=-100,
    )

    return {
        "images": images,
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "task_types": task_types,
        "prompt_texts": prompt_texts,
    }
