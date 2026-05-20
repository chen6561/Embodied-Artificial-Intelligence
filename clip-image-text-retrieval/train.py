import os
import random
import numpy as np
import torch
import torch.nn as nn
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from dataset import COCOCaptionDataset

# ==========================
# 训练超参数配置（可直接修改）
# ==========================
# 数据集路径
IMG_DIR = "D:/datasets/vla/coco/train2017"
ANN_FILE = "D:/datasets/vla/coco/annotations_trainval2017/annotations/captions_train2017.json"

# 训练硬件与批次
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# 训练轮次与学习率
EPOCHS = 20
LR = 1e-6

# 验证集比例与验证策略
VAL_RATIO = 0.05
VAL_EVERY_EPOCH = 1

# 模型保存策略
SAVE_BEST_ONLY = True

# ==========================
# 随机种子固定（保证实验可复现）
# ==========================
def set_random_seed(seed=42):
    """
    固定所有随机种子，确保训练可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 保证CuDNN确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed()

# ==========================
# 模型与优化器初始化
# ==========================
def build_model_and_optimizer():
    """
    加载CLIP模型并构建优化器
    返回：model, optimizer
    """
    # 加载CLIP官方预训练模型
    model, _ = clip.load("ViT-B/32", device=DEVICE)
    # 转为float32精度，避免半精度导致NaN/训练不稳定
    model = model.float()

    # 使用Adam优化器（CLIP微调标准配置）
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    return model, optimizer

# ==========================
# 数据集构建（训练集 + 验证集）
# ==========================
def build_dataloaders():
    """
    构建COCO标题数据集的训练/验证DataLoader
    返回：train_loader, val_loader
    """
    # 加载完整标注数据集
    full_dataset = COCOCaptionDataset(IMG_DIR, ANN_FILE)

    # 按比例划分训练集和验证集
    val_len = int(len(full_dataset) * VAL_RATIO)
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    # 训练集：打乱 + 训练模式
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )

    # 验证集：不打乱 + 评估模式
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )

    print(f"📦 数据集加载完成：训练集 {train_len} 张 | 验证集 {val_len} 张")
    return train_loader, val_loader

# ==========================
# 验证函数（计算检索精度）
# ==========================
@torch.no_grad()
def validate(model, val_loader, device):
    """
    在验证集上评估图像-文本双向检索精度
    返回：图搜文精度、文搜图精度、平均精度
    """
    # 切换模型为评估模式
    model.eval()

    # 统计变量初始化
    total_samples = 0
    correct_img2txt = 0
    correct_txt2img = 0

    # 逐批次验证
    for imgs, txts, _ in tqdm(val_loader, desc="Validating"):
        # 数据搬运到指定设备
        imgs = imgs.to(device).float()
        txts = txts.to(device)

        # CLIP前向推理，得到图像-文本、文本-图像相似度矩阵
        logits_img, logits_txt = model(imgs, txts)

        # 构造标签：batch内对角匹配（标准CLIP检索评估方式）
        labels = torch.arange(len(imgs), device=device)

        # 统计正确预测数量
        correct_img2txt += (logits_img.argmax(dim=1) == labels).sum().item()
        correct_txt2img += (logits_txt.argmax(dim=1) == labels).sum().item()

        total_samples += len(imgs)

    # 计算各类精度
    acc_img2txt = 100.0 * correct_img2txt / total_samples
    acc_txt2img = 100.0 * correct_txt2img / total_samples
    avg_acc = (acc_img2txt + acc_txt2img) / 2

    return acc_img2txt, acc_txt2img, avg_acc

# ==========================
# 模型保存函数
# ==========================
def save_best_checkpoint(epoch, best_acc, model, save_path="checkpoints/best_clip.pth"):
    """
    保存最优模型权重
    """
    # 自动创建保存目录
    os.makedirs("checkpoints", exist_ok=True)

    # 保存模型状态
    torch.save({
        "epoch": epoch,
        "acc": best_acc,
        "state_dict": model.state_dict(),
    }, save_path)

    print(f"💾 最优模型已保存：{save_path} | 精度：{best_acc:.2f}%")

# ==========================
# 训练主循环
# ==========================
def train_epoch(model, train_loader, optimizer, device, epoch):
    """
    单轮训练逻辑
    返回：本轮平均损失
    """
    model.train()  # 切换训练模式
    total_loss = 0.0

    # 进度条显示
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for i, (imgs, txts, _) in enumerate(pbar):
        # 数据搬移
        imgs = imgs.to(device).float()
        txts = txts.to(device)

        # CLIP 前向计算相似度
        logits_img, logits_txt = model(imgs, txts)

        # 构造标签
        labels = torch.arange(len(imgs), device=device)

        # 计算双向交叉熵损失
        loss_img = nn.CrossEntropyLoss()(logits_img, labels)
        loss_txt = nn.CrossEntropyLoss()(logits_txt, labels)
        loss = (loss_img + loss_txt) / 2  # 双向损失取平均

        # 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

        # 实时显示当前步loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 本轮平均损失
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# ==========================
# 主函数入口
# ==========================
def main():
    # 1. 初始化模型与优化器
    model, optimizer = build_model_and_optimizer()

    # 2. 构建数据加载器
    train_loader, val_loader = build_dataloaders()

    # 3. 开始训练
    best_acc = 0.0
    print("=" * 60)
    print("🚀 开始训练 CLIP 图像-文本检索模型")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # 训练一轮
        avg_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        print(f"✅ Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

        # 定期验证
        if (epoch + 1) % VAL_EVERY_EPOCH == 0:
            acc_img2txt, acc_txt2img, avg_acc = validate(model, val_loader, DEVICE)
            print(f"📊 验证结果：图搜文={acc_img2txt:.2f}% | 文搜图={acc_txt2img:.2f}% | 平均={avg_acc:.2f}%")

            # 保存最优模型
            if avg_acc > best_acc:
                best_acc = avg_acc
                save_best_checkpoint(epoch + 1, best_acc, model)

    # 训练结束
    print("\n🎉 训练完成！")
    print(f"🏆 最优验证精度: {best_acc:.2f}%")

if __name__ == '__main__':
    main()