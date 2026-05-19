#SigLIP（Sigmoid Loss for Language-Image Pre-training）是 CLIP 的改进版本，
#核心差异在于损失函数使用 Sigmoid 替代 InfoNCE（CrossEntropy），
#无需构造 batch 内的对比标签，训练更稳定且适合大批次场景。

# ======================== 导入依赖库 ========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from clip import tokenize  # 复用CLIP官方文本分词器

# ======================== 超参数定义 ========================
BATCH_SIZE = 16  # 批次大小
EPOCHS = 10  # 训练轮数
LR = 1e-4  # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型结构参数（与原CLIP保持一致）
VISION_DIM = 512
TEXT_DIM = 512
EMBED_DIM = 512
VOCAB_SIZE = 49408
TEXT_LEN = 77
IMAGE_SIZE = 224

# SigLIP特有超参数
TAU = 0.07  # 温度系数（固定值，SigLIP通常不使用可学习温度）
LABEL_SMOOTHING = 0.1  # 标签平滑，提升泛化性


# ======================== SigLIP模型定义 ========================
class SimpleSigLIP(nn.Module):
    def __init__(self):
        super().__init__()

        # 图像编码器（与原CLIP完全一致）
        self.image_encoder = nn.Sequential(
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE * 3, VISION_DIM),
            nn.ReLU(),
            nn.Linear(VISION_DIM, EMBED_DIM)
        )

        # 文本编码器（与原CLIP完全一致）
        self.text_encoder = nn.Sequential(
            nn.Embedding(VOCAB_SIZE, TEXT_DIM),
            nn.Flatten(),
            nn.Linear(TEXT_DIM * TEXT_LEN, EMBED_DIM)
        )

    # 图像特征提取（归一化）
    def encode_image(self, x):
        x = x.flatten(1)
        feat = self.image_encoder(x)
        return F.normalize(feat, dim=-1)

    # 文本特征提取（归一化）
    def encode_text(self, x):
        feat = self.text_encoder(x)
        return F.normalize(feat, dim=-1)

    # SigLIP前向传播
    def forward(self, image, text):
        # 提取图文特征
        image_feat = self.encode_image(image)  # [B, EMBED_DIM]
        text_feat = self.encode_text(text)  # [B, EMBED_DIM]

        # 计算图文相似度矩阵（缩放温度系数）
        # 与CLIP不同：SigLIP直接计算batch内所有图文对的相似度
        logits = (image_feat @ text_feat.t()) / TAU  # [B, B]

        return logits


# ======================== 生成训练数据（复用原逻辑） ========================
def generate_real_batch(batch_size):
    """生成模拟图像 + 真实英文句子token"""
    # 模拟图像 [B, 3, 224, 224]
    image = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    # 真实英文句子集合
    real_text_sentences = [
        "A cat is sitting on a couch.",
        "A dog runs across the grass.",
        "A beautiful sunset over the ocean.",
        "A group of people are playing football.",
        "A bird flies in the blue sky.",
        "A cup of coffee on the wooden table.",
        "A white car parked on the street.",
        "A small flower blooming in the garden.",
        "A book lies open on the desk.",
        "A child is eating ice cream happily.",
        "A mountain covered with snow.",
        "A boat floating on the lake.",
        "A man riding a bicycle in the park.",
        "A woman is taking photos with a camera.",
        "A bright moon in the night sky.",
        "A computer sits on a clean desk."
    ]

    selected = real_text_sentences[:batch_size]
    text_tokens = tokenize(selected).to(DEVICE)

    return image, text_tokens


# ======================== SigLIP损失函数（核心差异） ========================
def siglip_loss(logits):
    """
    SigLIP损失函数：Sigmoid Loss
    核心逻辑：
    1. 正样本（对角线）标签为1，负样本（非对角线）标签为0
    2. 使用带标签平滑的二元交叉熵（BCEWithLogitsLoss）
    3. 无需区分图像/文本视角，直接计算全局损失
    """
    batch_size = logits.shape[0]

    # 构造标签矩阵：对角线为正样本（1），其余为负样本（0）
    labels = torch.eye(batch_size, dtype=torch.float32).to(DEVICE)

    # 标签平滑：将1→(1-smoothing)，0→smoothing，通过 “软化” 标签，让模型从 “死记硬背” 转向 “学习本质特征”，提升泛化能力
    labels = labels * (1 - LABEL_SMOOTHING) + (1 - labels) * (LABEL_SMOOTHING / (batch_size - 1))

    # 计算带sigmoid的二元交叉熵损失（内置sigmoid，数值更稳定）
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    loss = loss_fn(logits, labels)

    return loss


# ======================== SigLIP训练主流程 ========================
def train_siglip():
    # 1. 初始化模型
    model = SimpleSigLIP().to(DEVICE)

    # 2. 定义优化器（沿用AdamW）
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"训练设备：{DEVICE}")
    print("使用【真实英文句子 + 模拟图像】开始训练SigLIP...\n")

    # 3. 多轮训练
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(range(100), desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for _ in pbar:
            # 获取训练数据
            image, text = generate_real_batch(BATCH_SIZE)

            # 前向传播：计算相似度矩阵
            logits = model(image, text)

            # 计算SigLIP损失
            loss = siglip_loss(logits)

            # 反向传播 & 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 打印本轮平均损失
        avg_loss = total_loss / 100
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}\n")

    # 保存模型
    torch.save(model.state_dict(), "simulated_siglip_real_text.pth")
    print("✅ 训练完成！模型已保存为 simulated_siglip_real_text.pth")


# ======================== 程序入口 ========================
if __name__ == "__main__":
    train_siglip()