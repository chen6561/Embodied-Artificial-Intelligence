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
EPOCHS = 100
LR = 1e-6

# 验证集比例与验证策略
VAL_RATIO = 0.05
VAL_EVERY_EPOCH = 1

# 模型保存策略
SAVE_BEST_ONLY = True

# SigLIP 特有超参数
TEMPERATURE = 0.07  # SigLIP 温度系数
SCALE_FACTOR = 20.0  # SigLIP 缩放因子（论文推荐值）


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
# SigLIP 核心计算函数
# ==========================
class SigLIPLoss(nn.Module):
    """SigLIP 损失计算模块（Sigmoid + Cross Entropy）"""

    def __init__(self, temperature=TEMPERATURE, scale_factor=SCALE_FACTOR):
        super().__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor

    def forward(self, image_embeds, text_embeds):
        # 归一化嵌入向量（和CLIP一致）
        image_embeds = nn.functional.normalize(image_embeds, dim=-1)
        text_embeds = nn.functional.normalize(text_embeds, dim=-1)

        # 计算相似度矩阵并缩放（SigLIP核心修改）
        logits = (image_embeds @ text_embeds.t()) * self.scale_factor / self.temperature

        # 构造batch内对角标签，原CLIP标签
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)

        # SigLIP使用BCEWithLogitsLoss（替代CLIP的CrossEntropy）
        # 构建正负样本掩码，SigLIP标签
        mask = torch.eye(batch_size, device=image_embeds.device)
        '''
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        '''

        # 图像到文本损失
        loss_img = nn.functional.binary_cross_entropy_with_logits(
            logits, mask, reduction='mean'
        )
        # 文本到图像损失
        loss_txt = nn.functional.binary_cross_entropy_with_logits(
            logits.t(), mask.t(), reduction='mean'
        )

        # 双向损失平均
        loss = (loss_img + loss_txt) / 2
        return loss, logits


# ==========================
# 模型与优化器初始化（替换为SigLIP逻辑）
# ==========================
def build_model_and_optimizer():
    """
    加载CLIP模型（权重复用），替换为SigLIP损失计算逻辑
    返回：model, optimizer, siglip_loss
    """
    # 加载CLIP官方预训练模型（权重复用，仅修改前向逻辑）
    model, _ = clip.load("ViT-B/32", device=DEVICE)
    # 转为float32精度，避免半精度导致NaN/训练不稳定
    model = model.float()

    # 使用Adam优化器（和CLIP微调配置一致）
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 初始化SigLIP损失函数
    siglip_loss = SigLIPLoss(temperature=TEMPERATURE, scale_factor=SCALE_FACTOR)

    return model, optimizer, siglip_loss


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
# 验证函数（计算检索精度，适配SigLIP输出）
# ==========================
@torch.no_grad()
def validate(model, val_loader, device, siglip_loss):
    """
    在验证集上评估图像-文本双向检索精度（适配SigLIP）
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

        # SigLIP前向推理：先获取嵌入向量，再计算相似度
        image_embeds = model.encode_image(imgs)
        text_embeds = model.encode_text(txts)

        # 计算相似度矩阵（复用SigLIP的缩放逻辑）
        logits = (image_embeds @ text_embeds.t()) * siglip_loss.scale_factor / siglip_loss.temperature

        # 构造标签：batch内对角匹配（标准检索评估方式）
        labels = torch.arange(len(imgs), device=device)

        # 统计正确预测数量
        correct_img2txt += (logits.argmax(dim=1) == labels).sum().item()
        correct_txt2img += (logits.t().argmax(dim=1) == labels).sum().item()

        total_samples += len(imgs)

    # 计算各类精度
    acc_img2txt = 100.0 * correct_img2txt / total_samples
    acc_txt2img = 100.0 * correct_txt2img / total_samples
    avg_acc = (acc_img2txt + acc_txt2img) / 2

    return acc_img2txt, acc_txt2img, avg_acc


# ==========================
# 模型保存函数
# ==========================
def save_best_checkpoint(epoch, best_acc, model, save_path="checkpoints/best_siglip.pth"):
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
        "temperature": TEMPERATURE,
        "scale_factor": SCALE_FACTOR
    }, save_path)

    print(f"💾 最优模型已保存：{save_path} | 精度：{best_acc:.2f}%")


# ==========================
# 训练主循环（适配SigLIP）
# ==========================
def train_epoch(model, train_loader, optimizer, siglip_loss, device, epoch):
    """
    单轮训练逻辑（替换为SigLIP损失计算）
    返回：本轮平均损失
    """
    model.train()  # 切换训练模式
    total_loss = 0.0

    # 进度条显示
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for i, (imgs, txts, _) in enumerate(pbar):
        # 数据搬移
        imgs = imgs.to(device).float()
        txts = txts.to(device)

        # SigLIP前向计算：先编码得到嵌入向量
        image_embeds = model.encode_image(imgs)
        text_embeds = model.encode_text(txts)

        # 计算SigLIP损失
        loss, _ = siglip_loss(image_embeds, text_embeds)

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
    # 1. 初始化模型、优化器和SigLIP损失函数
    model, optimizer, siglip_loss = build_model_and_optimizer()

    # 2. 构建数据加载器
    train_loader, val_loader = build_dataloaders()

    # 3. 开始训练
    best_acc = 0.0
    print("=" * 60)
    print("🚀 开始训练 SigLIP 图像-文本检索模型")
    print(f"📌 温度系数: {TEMPERATURE} | 缩放因子: {SCALE_FACTOR}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # 训练一轮（传入SigLIP损失函数）
        avg_loss = train_epoch(model, train_loader, optimizer, siglip_loss, DEVICE, epoch)
        print(f"✅ Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

        # 定期验证
        if (epoch + 1) % VAL_EVERY_EPOCH == 0:
            acc_img2txt, acc_txt2img, avg_acc = validate(model, val_loader, DEVICE, siglip_loss)
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