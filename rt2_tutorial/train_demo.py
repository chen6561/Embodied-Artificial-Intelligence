from __future__ import annotations

import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from rt2_model import RT2Config, RT2Model
from rt2_tokenizer import RT2Tokenizer
from synthetic_data import MixedRT2Dataset, build_language_model_batch


def train_one_epoch(
    model: RT2Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    训练一个 epoch，并返回平均 loss。
    """

    model.train()
    total_loss = 0.0
    total_steps = 0

    batch_fetch_start = time.perf_counter()
    for step_idx, batch in enumerate(loader, start=1):
        batch_fetch_end = time.perf_counter()

        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        forward_start = time.perf_counter()
        outputs = model(images=images, input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        forward_end = time.perf_counter()

        backward_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        backward_end = time.perf_counter()

        step_start = time.perf_counter()
        optimizer.step()
        step_end = time.perf_counter()

        total_loss += float(loss.item())
        total_steps += 1

        # 每个 step 都打印状态和计时，方便定位卡点到底在：
        # 1. DataLoader 取 batch
        # 2. forward
        # 3. backward
        # 4. optimizer.step
        print(
            f"[train] step={step_idx:03d} "
            f"loss={loss.item():.4f} "
            f"fetch={batch_fetch_end - batch_fetch_start:.3f}s "
            f"forward={forward_end - forward_start:.3f}s "
            f"backward={backward_end - backward_start:.3f}s "
            f"step={step_end - step_start:.3f}s "
            f"batch_images={tuple(images.shape)} "
            f"batch_input_ids={tuple(input_ids.shape)}"
        )

        # 下一轮开始前，记录“等待下一个 batch”的起点。
        batch_fetch_start = time.perf_counter()

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def run_vqa_demo(
    model: RT2Model,
    tokenizer: RT2Tokenizer,
    dataset: MixedRT2Dataset,
    device: torch.device,
) -> None:
    """
    跑一个 VQA 样例，展示“同一个模型可以像 VLM 一样回答文本问题”。
    """

    # 找一条偶数索引样本，保证它来自 VQA 任务。
    sample = dataset[0]
    images = sample["images"].unsqueeze(0).to(device)  # type: ignore[union-attr]
    prompt_ids = sample["prompt_ids"].tolist()  # type: ignore[union-attr]

    generated = model.generate(
        images=images,
        prompt_ids=[tokenizer.bos_token_id] + prompt_ids,
        max_new_tokens=4,
        constrained_token_ids=None,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("\n[VQA 示例]")
    print("prompt:", sample["prompt_text"])
    print("generated tokens:", tokenizer.decode_text(generated))


@torch.no_grad()
def run_robot_demo(
    model: RT2Model,
    tokenizer: RT2Tokenizer,
    dataset: MixedRT2Dataset,
    device: torch.device,
) -> None:
    """
    跑一个机器人动作生成样例，展示“动作 token 约束解码”。
    """

    # 找一条奇数索引样本，保证它来自机器人任务。
    sample = dataset[1]
    images = sample["images"].unsqueeze(0).to(device)  # type: ignore[union-attr]
    prompt_ids = sample["prompt_ids"].tolist()  # type: ignore[union-attr]

    generated = model.generate(
        images=images,
        prompt_ids=[tokenizer.bos_token_id] + prompt_ids,
        max_new_tokens=tokenizer.num_action_dims + 1,
        constrained_token_ids=tokenizer.robot_action_constraints(include_eos=True),
        eos_token_id=tokenizer.eos_token_id,
    )

    action_values = tokenizer.decode_action_token_ids(generated)

    print("\n[Robot 示例]")
    print("prompt:", sample["prompt_text"])
    print("generated raw tokens:", tokenizer.decode_text(generated, skip_special_tokens=False))
    print("decoded action:", [round(float(x), 3) for x in action_values.tolist()])


def main() -> None:
    """
    教学版 RT-2 训练入口。

    运行逻辑：
    1. 构造统一 tokenizer
    2. 构造混合数据集（VQA + Robot）
    3. 训练一个小型多模态自回归 Transformer
    4. 分别演示文本回答和动作生成
    """

    config = RT2Config()
    tokenizer = RT2Tokenizer(num_action_dims=8, action_bins=64)

    # 把 tokenizer 词表大小写回模型配置，保证 embedding 维度一致。
    config.vocab_size = tokenizer.vocab_size

    print("RT-2 教学版配置:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print(f"  tokenizer_vocab_size: {tokenizer.vocab_size}")

    dataset = MixedRT2Dataset(
        tokenizer=tokenizer,
        num_samples=512,
        use_chain_of_thought=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: build_language_model_batch(batch, tokenizer),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RT2Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    print(f"\n训练设备: {device}")
    for epoch in range(3):
        average_loss = train_one_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch + 1} average_loss={average_loss:.4f}")

    run_vqa_demo(model, tokenizer, dataset, device)
    run_robot_demo(model, tokenizer, dataset, device)


if __name__ == "__main__":
    main()
