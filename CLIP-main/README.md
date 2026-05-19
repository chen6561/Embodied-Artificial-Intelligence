# SigLIP & CLIP 模拟训练实现

本项目提供了 **CLIP**（Contrastive Language-Image Pre-training）和其改进版本 **SigLIP**（Sigmoid Loss for Language-Image Pre-training）的极简模拟训练实现，核心聚焦两者的核心差异（损失函数），使用模拟图像+真实英文句子完成端到端训练。

## 核心差异
| 特性                | CLIP                          | SigLIP                        |
|---------------------|-------------------------------|-------------------------------|
| 损失函数            | InfoNCE（CrossEntropy）| Sigmoid Loss（BCEWithLogits） |
| 温度系数            | 可学习参数                    | 固定值（无需学习）|
| 损失计算逻辑        | 图像/文本双视角对比损失       | 全局图文对二元交叉熵         |
| 标签构造            | 对角线为正确标签（0~B-1）| 对角线为1，其余为0（带平滑） |
| 优势                | 经典图文对比学习基线          | 训练更稳定、适合大批次场景    |

## 环境依赖
```bash
# 基础依赖
pip install torch tqdm clip
# 注意：clip库需安装官方版本（openai/CLIP），用于文本tokenize
```

## 快速开始

### 1. 训练CLIP
```bash
python train_clip_simulated.py
```

### 2. 训练SigLIP
```bash
python train_siglip_simulated.py
```

## 代码结构说明

### 通用模块
| 模块                | 功能说明                                                                 |
|---------------------|--------------------------------------------------------------------------|
| 超参数定义          | 批次大小、学习率、特征维度等（与官方CLIP保持一致）|
| 数据生成函数        | 生成随机模拟图像（3×224×224）+ 真实英文句子token（CLIP官方分词器处理）|
| 编码器结构          | 图像/文本编码器均采用极简线性层实现（保持CLIP特征维度对齐逻辑）|

### CLIP 核心实现
- **模型类**：`SimpleCLIP`，包含可学习温度系数
- **前向传播**：计算图像→文本、文本→图像双向相似度矩阵
- **损失函数**：`clip_loss`，双视角交叉熵损失平均（InfoNCE）

### SigLIP 核心实现
- **模型类**：`SimpleSigLIP`，温度系数固定（无需学习）
- **前向传播**：仅计算全局图文相似度矩阵（无需双向拆分）
- **损失函数**：`siglip_loss`，带标签平滑的二元交叉熵（Sigmoid Loss）

## 关键超参数说明
| 参数名             | 取值    | 说明                                                                 |
|--------------------|---------|----------------------------------------------------------------------|
| BATCH_SIZE         | 16      | 批次大小（每次训练16组图文对）|
| EPOCHS             | 10      | 训练轮数（每轮训练100个batch）|
| LR                 | 1e-4    | 学习率（AdamW优化器）|
| TAU（SigLIP）| 0.07    | 固定温度系数（SigLIP标准取值）|
| LABEL_SMOOTHING    | 0.1     | 标签平滑系数（提升SigLIP泛化性）|
| EMBED_DIM          | 512     | 图文特征最终对齐维度（与CLIP官方一致）|

## 训练输出
- 训练过程实时显示每batch的损失值，每轮结束打印平均损失
- 训练完成后自动保存模型权重：
  - CLIP：`simulated_clip_real_text.pth`
  - SigLIP：`simulated_siglip_real_text.pth`

## 核心逻辑亮点
1. **极简实现**：用线性层替代CLIP复杂的视觉/文本Transformer，聚焦核心对比逻辑
2. **数据模拟**：无需真实图像数据集，随机张量模拟图像+真实英文句子保证文本合理性
3. **损失对比**：清晰区分CLIP双视角交叉熵 vs SigLIP全局Sigmoid Loss
4. **标签平滑**：SigLIP实现标签平滑逻辑，避免模型过拟合到硬标签

## 注意事项
1. 本实现为**教学演示版**，仅保留核心逻辑，未使用CLIP官方的Transformer编码器
2. 温度系数：CLIP为可学习参数（初始值2.6592），SigLIP固定为0.07（行业标准）
3. 硬件支持：自动检测CUDA，优先使用GPU训练（CPU也可运行，速度较慢）
4. 文本处理：复用CLIP官方tokenizer，保证文本输入格式与官方一致

## 扩展方向
1. 替换编码器：将线性层替换为CLIP官方的Vision Transformer/Text Transformer
2. 真实数据：接入COCO、Flickr等图文数据集，替换模拟图像
3. 大批次训练：验证SigLIP在大批次下的训练稳定性优势
4. 温度系数对比：测试SigLIP使用可学习温度系数的效果