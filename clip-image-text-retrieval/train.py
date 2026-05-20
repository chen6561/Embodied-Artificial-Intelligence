import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import COCOCaptionDataset
import clip
import os

# ==================
# 训练配置（你可以随便改）
# ==================
IMG_DIR = "D:/datasets/vla/coco/train2017"
ANN_FILE = "D:/datasets/vla/coco/annotations_trainval2017/annotations/captions_train2017.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-5
VAL_RATIO = 0.05  # 验证集比例 5%
VAL_EVERY_EPOCH = 1  # 每 1 轮验证一次
SAVE_BEST_ONLY = True  # 只保最好的模型

# ==================
# 随机种子（保证可复现）
# ==================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed()

# ==================
# 模型 & 优化器
# ==================
model, _ = clip.load("ViT-B/32", device=DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==================
# 数据集：自动切分 训练集 / 验证集
# ==================
dataset = COCOCaptionDataset(IMG_DIR, ANN_FILE)
val_len = int(len(dataset) * VAL_RATIO)
train_len = len(dataset) - val_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

# 训练 Dataloader
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
# 验证 Dataloader
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ==================
# 验证函数（计算图文检索准确率）
# ==================
@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total = 0
    correct_img2txt = 0
    correct_txt2img = 0

    for img, txt, _ in tqdm(val_loader, desc="🔍 Validating"):
        img = img.to(device)
        txt = txt.to(device)

        logits_img, logits_txt = model(img, txt)
        probs = logits_img.softmax(dim=-1)
        labels = torch.arange(len(img)).to(device)

        # 图像→文本 召回率
        correct_img2txt += (probs.argmax(dim=1) == labels).sum().item()

        # 文本→图像 召回率
        probs_txt = logits_txt.softmax(dim=-1)
        correct_txt2img += (probs_txt.argmax(dim=1) == labels).sum().item()

        total += len(img)

    acc_img2txt = 100 * correct_img2txt / total
    acc_txt2img = 100 * correct_txt2img / total
    avg_acc = (acc_img2txt + acc_txt2img) / 2
    return acc_img2txt, acc_txt2img, avg_acc

# ==================
# 保存模型
# ==================
def save_checkpoint(epoch, acc, path="best_clip.pth"):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save({
        "epoch": epoch,
        "acc": acc,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, path)
    print(f"💾 Model saved to {path}")

# ==================
# 训练主函数
# ==================
def main():
    best_acc = 0.0
    print("🚀 训练 CLIP on COCO 数据集")
    print(f"📊 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # 训练
        for img, txt, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            img = img.to(DEVICE)
            txt = txt.to(DEVICE)

            logits_img, logits_txt = model(img, txt)
            labels = torch.arange(len(img)).to(DEVICE)

            loss = (
                nn.CrossEntropyLoss()(logits_img, labels) +
                nn.CrossEntropyLoss()(logits_txt, labels)
            ) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # 验证
        if (epoch + 1) % VAL_EVERY_EPOCH == 0:
            acc_img2txt, acc_txt2img, avg_acc = validate(model, val_loader, DEVICE)
            print(f"📊 验证结果：")
            print(f"   图像→文本 召回率: {acc_img2txt:.2f}%")
            print(f"   文本→图像 召回率: {acc_txt2img:.2f}%")
            print(f"   平均准确率: {avg_acc:.2f}%")

            # 保存最优模型
            if SAVE_BEST_ONLY:
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    save_checkpoint(epoch+1, best_acc, "checkpoints/best_clip.pth")
            else:
                save_checkpoint(epoch+1, avg_acc, f"checkpoints/clip_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    main()