# ======================== 导入依赖库 ========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # 用于显示训练进度条，让训练过程更直观
from clip import tokenize  # 导入CLIP官方的文本分词器，用于处理真实英文句子

# ======================== 超参数定义 ========================
# 超参数：训练前手动设定的参数，控制训练速度、模型大小、显存占用等
BATCH_SIZE = 16          # 批次大小：每次训练输入16组图文对
EPOCHS = 10              # 训练轮数：将所有模拟数据完整训练10轮
LR = 1e-4                # 学习率：控制模型参数更新的步长，步长过大不收敛，过小训练慢
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择GPU（cuda）或CPU

# CLIP模型结构参数（固定，保证图像和文本特征维度对齐）
VISION_DIM = 512         # 图像编码器输出的特征维度
TEXT_DIM = 512           # 文本编码器输出的特征维度
EMBED_DIM = 512          # 最终对齐的特征维度（图像、文本必须一致）
VOCAB_SIZE = 49408       # CLIP官方真实词表大小（固定值，不可随意修改）
TEXT_LEN = 77            # CLIP固定文本长度：所有句子都会被填充/截断为77个token
IMAGE_SIZE = 224         # CLIP标准输入图像尺寸：224x224

# ======================== 简易CLIP模型定义 ========================
class SimpleCLIP(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------------------- 图像编码器 ----------------------
        # 功能：将一张图像 → 压缩成一个固定维度的特征向量
        # 让模型能够“理解”图像内容
        self.image_encoder = nn.Sequential(
            # 输入：展平后的图像 3×224×224 → 输出512维特征
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE * 3, VISION_DIM),
            nn.ReLU(),  # 激活函数，增加模型非线性拟合能力
            nn.Linear(VISION_DIM, EMBED_DIM)
        )

        # ---------------------- 文本编码器 ----------------------
        # 功能：将一段英文句子 → 压缩成和图像同维度的特征向量
        # 实现图文特征空间对齐
        self.text_encoder = nn.Sequential(
            # Embedding层：将文本token（数字）转换为稠密向量
            nn.Embedding(VOCAB_SIZE, TEXT_DIM),
            nn.Flatten(),  # 将多维特征展平为一维
            nn.Linear(TEXT_DIM * TEXT_LEN, EMBED_DIM)
        )

        # ---------------------- CLIP温度系数（可学习参数） ----------------------
        # 专业名称：logit_scale / 温度系数
        # 作用：自动缩放图文相似度分数，让训练更稳定、softmax更容易区分正负样本
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    # ---------------------- 图像特征提取 ----------------------
    def encode_image(self, x):
        # 将图像展平：[B,3,224,224] → [B, 3*224*224]
        x = x.flatten(1)
        feat = self.image_encoder(x)
        # 特征归一化：让所有特征向量长度=1，只保留方向差异，便于计算余弦相似度
        return F.normalize(feat, dim=-1)

    # ---------------------- 文本特征提取 ----------------------
    def encode_text(self, x):
        feat = self.text_encoder(x)
        # 同样做归一化，保证图文特征计算方式一致
        return F.normalize(feat, dim=-1)

    # ---------------------- CLIP前向传播（核心） ----------------------
    def forward(self, image, text):
        # 1. 分别提取图像特征和文本特征
        image_feat = self.encode_image(image)
        text_feat = self.encode_text(text)

        # 2. 温度系数：取指数，保证缩放系数为正数
        logit_scale = self.logit_scale.exp()

        # 3. 计算图文相似度矩阵
        # @ ：矩阵乘法 → 表示计算两两之间的相似度
        # .t() ：矩阵转置 → 让文本特征矩阵形状匹配，才能做矩阵乘法
        logits_per_image = logit_scale * image_feat @ text_feat.t()

        # logits_per_image 是图像→文本的相似度矩阵，结构如下：
        #          文本1  文本2  文本3 ...
        # 图1      0.9    0.1    0.0
        # 图2      0.1    0.8    0.1
        # 图3      0.0    0.2    0.9
        # ...      ...    ...    ...
        # 矩阵中数值越大，代表对应图文对的匹配度越高

        # 对图像-文本相似度矩阵进行转置（行列互换）
        # 转置后 logits_per_text 是文本→图像的相似度矩阵，结构如下：
        #          图1   图2   图3 ...
        # 文本1    0.9   0.1   0.0
        # 文本2    0.1   0.8   0.2
        # 文本3    0.0   0.1   0.9
        # ...      ...   ...   ...
        # 目的：从文本视角计算损失，让CLIP从图像、文本两个方向学习图文对齐

        # 4. 转置得到文本→图像的相似度矩阵
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

# ======================== 生成模拟图像 + 真实英文句子 ========================
def generate_real_batch(batch_size):
    """
    生成一批训练数据：
    图像：随机模拟生成（无需真实图片）
    文本：真实英文句子（更贴近CLIP实际训练场景）
    """
    # 生成模拟图像：形状 [B, 3, 224, 224]
    # B=批次，3=RGB三通道，224×224=图像尺寸
    image = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    # 真实英文句子集合（模拟图文对中的文本）
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

    # 从句子列表中选取当前批次数量的句子
    selected = real_text_sentences[:batch_size]

    # 使用CLIP官方tokenizer将英文句子转为模型可接受的token序列
    text_tokens = tokenize(selected).to(DEVICE)

    return image, text_tokens

# ======================== CLIP对比损失函数（InfoNCE） ========================
def clip_loss(logits_per_image, logits_per_text):
    """
    对比损失函数：
    让【匹配的图文对】相似度最高
    让【不匹配的图文对】相似度最低
    """
    batch_size = logits_per_image.shape[0]
    # 构造对比学习的标签
    # 在一个batch中，第 i 张图像 对应 第 i 段文本
    # 正确匹配的图文对都在【对角线】上，因此标签就是 0,1,2,...,batch_size-1
    labels = torch.arange(batch_size).to(DEVICE)

    # 从图像视角计算损失：每张图像应匹配自己的文本
    loss_img = F.cross_entropy(logits_per_image, labels)
    # 从文本视角计算损失：每段文本应匹配自己的图像
    loss_txt = F.cross_entropy(logits_per_text, labels)

    # 总损失为两者平均值，保证双向对齐
    return (loss_img + loss_txt) / 2

# ======================== 训练主流程 ========================
def train():
    # 1. 初始化模型并移至设备（GPU/CPU）
    model = SimpleCLIP().to(DEVICE)

    # 2. 定义优化器：AdamW（CLIP官方使用的优化器）
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"训练设备：{DEVICE}")
    print("使用【真实英文句子 + 模拟图像】开始训练CLIP...\n")

    # 3. 开始多轮训练
    for epoch in range(EPOCHS):
        model.train()  # 切换为训练模式（启用dropout、batchnorm等训练特性）
        total_loss = 0.0

        # 每轮训练100个batch
        pbar = tqdm(range(100), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for _ in pbar:
            # 获取模拟图像 + 真实英文句子token
            image, text = generate_real_batch(BATCH_SIZE)

            # 前向传播：计算图文相似度
            logits_img, logits_txt = model(image, text)

            # 计算对比损失
            loss = clip_loss(logits_img, logits_txt)

            # 反向传播 & 更新参数
            optimizer.zero_grad()  # 清空上一步的梯度，避免累积
            loss.backward()        # 反向传播，计算梯度
            optimizer.step()       # 根据梯度更新模型权重

            # 记录损失并显示
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 打印本轮平均损失
        avg_loss = total_loss / 100
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}\n")

    # 训练结束，保存模型权重
    torch.save(model.state_dict(), "simulated_clip_real_text.pth")
    print("✅ 训练完成！模型已保存为 simulated_clip_real_text.pth")

# ======================== 程序入口 ========================
if __name__ == "__main__":
    train()