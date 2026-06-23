from dataclasses import asdict

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from rt1_model import RT1Config, RT1Model, discretize_action, undiscretize_action


class ToyRT1Dataset(Dataset):
    """
    一个合成数据集，用来验证模型结构和训练流程是否合理。

    这里的动作 token 是图像序列和文本的确定性函数，因此任务不至于太简单，
    但又足够容易，让这个教学版模型能学到东西。

    为什么要用合成数据：
    这个脚本的目标是演示 RT-1 风格模型的接线方式和训练闭环，
    而不是依赖真实机器人数据集。于是我们人为构造一个同时依赖图像和文本的标签，
    这样模型不能靠记忆常数标签蒙混过关。
    """

    def __init__(self, config: RT1Config, num_samples: int = 256) -> None:
        self.config = config
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        # 用样本索引作为固定随机种子，使数据集是可重复的。
        # 这样调试时，同一个 index 总会返回同一条样本。
        g = torch.Generator().manual_seed(index)

        # 伪造的多帧 RGB 观测序列：
        # [time, 3, H, W]
        images = torch.rand(
            self.config.sequence_length,
            3,
            self.config.image_size,
            self.config.image_size,
            generator=g,
        )

        # 伪造的文本指令嵌入。
        # 在真实项目里，这一步应该来自语言编码器。
        text = torch.randn(self.config.text_dim, generator=g)

        # 构造一个同时依赖视觉和语言的目标动作。
        # 做法是：
        # 1. 对每帧做空间平均；
        # 2. 再对时间维做汇总；
        # 3. 与文本统计量混合；
        # 这样得到一个稳定但并不平凡的 input -> action 映射，
        # 便于 RT-1 风格模型学习。
        frame_stats = images.mean(dim=(2, 3))
        # 原来的 temporal_stats 只有 3 维，因为它只是对 RGB 三个通道做了时间平均。
        # 但动作目标有 num_action_dims 维，所以我们把 [time, 3] 展平后再分块，
        # 得到与动作维度一致的视觉统计量。
        visual_summary = frame_stats.reshape(-1)
        # 这里使用 tensor_split 而不是 chunk。
        # chunk 在“长度不能理想整分”时，不一定严格返回指定数量的块；
        # tensor_split 会稳定返回精确的 num_action_dims 份。
        visual_chunks = torch.tensor_split(visual_summary, self.config.num_action_dims)
        temporal_stats = torch.stack([chunk.mean() for chunk in visual_chunks])
        # 这里不能直接把长度为 text_dim 的向量 reshape 成
        # [num_action_dims, -1]，因为 text_dim 未必能被动作维度数整除。
        # 更稳妥的做法是把文本向量切成 num_action_dims 份不等长小块，
        # 再分别求均值，最终得到每个动作维度对应的一个文本统计量。
        text_chunks = torch.tensor_split(text, self.config.num_action_dims)
        text_stats = torch.stack([chunk.mean() for chunk in text_chunks])
        continuous_action = torch.tanh(
            0.7 * temporal_stats + 0.3 * text_stats
        )

        # 把连续动作值离散成 token bin。
        action_tokens = discretize_action(continuous_action, self.config.vocab_size)
        return images, text, action_tokens


def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    对所有动作维度统一计算交叉熵损失。

    logits: [batch, action_dims, vocab_size]
    targets: [batch, action_dims]
    """

    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def main() -> None:
    # 1. 构建配置并打印，方便确认当前实验设定。
    config = RT1Config()
    print("Config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    # 2. 创建一个小型合成数据集和对应的 DataLoader。
    dataset = ToyRT1Dataset(config, num_samples=384)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 3. 常规的 device / model / optimizer 初始化。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RT1Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    print(f"Training on: {device}")
    model.train()

    # 4. 一个最小可运行训练循环。
    # 这里只训练几个 epoch，因为目标是演示结构，不是追求性能指标。
    for epoch in range(3):
        running_loss = 0.0
        for step, (images, text, target_tokens) in enumerate(loader, start=1):
            # 把当前 mini-batch 移到指定设备。
            images = images.to(device)
            text = text.to(device)
            target_tokens = target_tokens.to(device)

            # 前向传播：
            # images -> 压缩后的视觉 token -> Transformer -> 动作 logits
            logits = model(images, text)
            loss = compute_loss(logits, target_tokens)

            # 反向传播和参数更新。
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 10 == 0:
                # 每 10 个 step 打印一次平均 loss，读起来更平滑。
                avg_loss = running_loss / 10
                print(f"epoch={epoch + 1} step={step:03d} loss={avg_loss:.4f}")
                running_loss = 0.0

    # 5. 用一条样本做推理，作为一个便于肉眼检查的 sanity check。
    model.eval()
    with torch.no_grad():
        images, text, target_tokens = dataset[0]
        logits = model(images.unsqueeze(0).to(device), text.unsqueeze(0).to(device))
        predicted_tokens = logits.argmax(dim=-1).squeeze(0).cpu()
        predicted_action = undiscretize_action(predicted_tokens, config.vocab_size)
        target_action = undiscretize_action(target_tokens, config.vocab_size)

    print("\nSample prediction")
    print("target tokens   :", target_tokens.tolist())
    print("predicted tokens:", predicted_tokens.tolist())
    # 把离散 token 还原成近似连续值后，更容易直观看出预测是否接近目标。
    print("target action   :", [round(x, 3) for x in target_action.tolist()])
    print("predicted action:", [round(x, 3) for x in predicted_action.tolist()])


if __name__ == "__main__":
    main()
