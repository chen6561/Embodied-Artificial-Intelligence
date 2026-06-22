import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
# 请确保 dataset.py 在同级目录
from dataset import COCOCaptionDataset

# ==========================
# 全局推理配置（与训练保持一致）
# ==========================
# 数据集路径配置
IMG_DIR = "D:/datasets/vla/coco/train2017"
ANN_FILE = "D:/datasets/vla/coco/annotations_trainval2017/annotations/captions_train2017.json"

# 设备与批次配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# 微调后模型权重路径
CHECKPOINT_PATH = "checkpoints/best_clip.pth"

def load_trained_clip_model():
    """
    加载预训练CLIP模型 + 加载微调后的权重
    返回：model, preprocess（图像预处理）
    """
    # 加载官方CLIP模型结构
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model = model.float()  # 切换为float32精度，避免半精度报错

    # 加载微调后的模型权重
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    # 打印加载信息
    print(f"✅ 模型加载完成")
    print(f"📦 最优轮次: {checkpoint['epoch']}")
    print(f"🎯 最优精度: {checkpoint['acc']:.2f}%")

    # 切换为评估模式（禁用dropout、bn等）
    model.eval()
    return model, preprocess

@torch.no_grad()
def evaluate_full_validation_set(model):
    """
    在完整COCO验证集上计算检索精度
    评估指标：图搜文 / 文搜图 / 平均精度
    """
    # 构建数据集与数据加载器
    dataset = COCOCaptionDataset(IMG_DIR, ANN_FILE)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )

    # 统计变量初始化
    total_samples = 0
    correct_img2txt = 0
    correct_txt2img = 0

    # 逐批次推理
    for images, texts, _ in tqdm(data_loader, desc="Evaluating..."):
        # 数据搬移到设备
        images = images.to(DEVICE, non_blocking=True).float()
        texts = texts.to(DEVICE, non_blocking=True)

        # CLIP 前向推理
        logits_per_image, logits_per_text = model(images, texts)

        # 标签：batch 内对角匹配（标准CLIP检索评估方式）
        batch_labels = torch.arange(len(images), device=DEVICE)

        # 统计正确数
        correct_img2txt += (logits_per_image.argmax(dim=1) == batch_labels).sum().item()
        correct_txt2img += (logits_per_text.argmax(dim=1) == batch_labels).sum().item()

        total_samples += len(images)

    # 计算精度
    acc_img2txt = 100.0 * correct_img2txt / total_samples
    acc_txt2img = 100.0 * correct_txt2img / total_samples
    avg_acc = (acc_img2txt + acc_txt2img) / 2

    # 打印结果
    print("\n" + "=" * 60)
    print("📊 完整验证集检索精度评估结果")
    print(f"图搜文 准确率: {acc_img2txt:.2f}%")
    print(f"文搜图 准确率: {acc_txt2img:.2f}%")
    print(f"平均检索精度: {avg_acc:.2f}%")
    print("=" * 60)

    return acc_img2txt, acc_txt2img, avg_acc

@torch.no_grad()
def single_image_retrieval_demo(model, preprocess):
    """
    单张图片 + 多个文本句子的相似度检索 Demo
    可直接替换图片路径与句子测试
    """
    # 测试图片路径（可自行修改）
    image_path = "demo.png"

    # 候选描述文本（可自行增删、修改）
    text_candidates = [
        "a dog playing in the grass",
        "a cat sitting on a sofa",
        "a person riding a bike"
    ]

    # 图像预处理
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # 文本编码
    text_tokens = clip.tokenize(text_candidates).to(DEVICE)

    # 推理相似度
    logits_per_image, _ = model(image_tensor, text_tokens)
    probabilities = logits_per_image.softmax(dim=-1).cpu().numpy().squeeze()

    # 输出结果
    print("\n🔎 单张图像检索演示结果")
    for idx, sentence in enumerate(text_candidates):
        print(f"[{idx+1}] {sentence}")
        print(f"    相似度概率: {probabilities[idx]:.4f}\n")

def main():
    """主函数：统一调度模型加载、评估、演示"""
    # 1. 加载模型
    model, preprocess = load_trained_clip_model()

    # 2. 完整验证集评估（取消注释即可运行）
    # evaluate_full_validation_set(model)

    # 3. 单张图片检索演示
    single_image_retrieval_demo(model, preprocess)

if __name__ == "__main__":
    main()