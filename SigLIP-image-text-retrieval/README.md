# CLIP 图文检索微调与推理框架
基于 COCO Caption 数据集构建的 CLIP 图像-文本双向检索训练、验证、推理全套工程，支持**微调训练**、**双向检索精度评估**、**单图demo推理**，代码规范、注释完整、开箱即用。

---

## 📁 项目结构
```
├── dataset.py          # COCO 标题数据集加载与预处理
├── train.py            # CLIP 模型微调训练主脚本
├── inference.py        # 模型评估 + 单图检索推理
├── utils.py            # 通用工具函数（评估函数）
├── checkpoints/        # 模型保存目录（自动创建）
└── README.md           # 项目说明文档
```

---

## 🧠 项目介绍
本项目基于 **OpenAI CLIP (ViT-B/32)** 实现：
- 在 COCO Caption 数据集上进行**图像-文本双向检索微调**
- 支持**图搜文 / 文搜图**双向精度评估
- 训练过程自动保存最优模型
- 提供单张图片检索 Demo，可直接测试效果
- 代码遵循 PyTorch 工程规范，注释详细

---

## 🚀 快速开始

### 1. 环境依赖
```bash
pip install torch torchvision clip-openai tqdm pillow numpy
```

### 2. 数据集准备
将 COCO 数据集放在如下路径（可在代码中修改）：
```
# 图片目录
D:/datasets/vla/coco/train2017/

# 标注文件
D:/datasets/vla/coco/annotations_trainval2017/annotations/captions_train2017.json
```

---

## 🔧 使用说明

### 1. 训练 CLIP 模型
```bash
python train.py
```
**训练功能：**
- 自动划分训练集/验证集（5% 验证）
- 双向交叉熵损失训练
- 梯度裁剪防止爆炸
- 每轮验证并只保存最优模型
- 输出图搜文、文搜图、平均精度

**训练配置（train.py 顶部可修改）：**
```python
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-6
VAL_RATIO = 0.05
```

---

### 2. 模型评估 & 推理
```bash
python inference.py
```
**支持两种模式：**
1. **完整验证集精度评估**（取消注释即可运行）
2. **单张图片检索 Demo**（默认开启）

**输出示例：**
```
✅ 模型加载完成
📦 最优轮次: 8
🎯 最优精度: 89.62%

🔎 单张图像检索演示结果
[1] a dog playing in the grass
    相似度概率: 0.9241
...
```

---

### 3. 数据集测试
检查数据集是否正常加载：
```bash
python dataset.py
```
输出图像 shape、文本 token shape、标题文本。

---

## 📊 评估指标
模型采用 **CLIP 标准检索评估方式**：
- **图搜文 Acc**：图像特征检索对应文本正确率
- **文搜图 Acc**：文本特征检索对应图像正确率
- **平均精度**：(图搜文 + 文搜图) / 2

评估逻辑：**batch 内对角匹配**（第 i 张图 ↔ 第 i 个文本）

---

## 📌 核心文件说明

### dataset.py
COCO caption 数据集加载类，实现：
- JSON 标注兼容解码（UTF-8 / latin-1）
- 图像统一 resize 224×224 + 标准化
- CLIP 文本 tokenize
- 图片-标题对构建

### train.py
训练主流程：
- 固定随机种子确保可复现
- 加载 CLIP 并转换为 float32
- 构建训练/验证 DataLoader
- 双向损失训练 + 最优模型保存

### inference.py
推理与评估：
- 加载微调后模型权重
- 全验证集精度计算
- 单图 + 多文本相似度检索 demo

### utils.py
工具函数：
- `evaluate_clip()`：单方向图搜文精度评估

---

## 🎯 训练结果示例
```
✅ Epoch 8 平均损失: 0.1245
📊 验证结果：图搜文=88.54% | 文搜图=90.70% | 平均=89.62%
💾 最优模型已保存：checkpoints/best_clip.pth | 精度：89.62%
```

---

## ⚠️ 注意事项
1. 路径请根据自己机器修改 `IMG_DIR` / `ANN_FILE`
2. 训练建议使用 **GPU**，CPU 速度较慢
3. 若出现 NaN，代码已默认使用 `model.float()` 修复
4. 微调学习率建议 **1e-6 ~ 1e-5**
5. 推理时请确保使用 **训练相同的预处理**

---

## ✨ 扩展功能（可自行添加）
- 多卡训练
- TensorBoard / WandB 日志
- 批量图像检索
- 检索结果可视化
- 更多数据集适配

---
