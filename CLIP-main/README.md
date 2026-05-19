# SigLIP & CLIP & VLA 模拟训练实现

本项目提供了 **CLIP**（Contrastive Language-Image Pre-training）、其改进版本 **SigLIP**（Sigmoid Loss for Language-Image Pre-training）以及 **VLA（Vision-Language-Action）** 视觉-语言-动作模型的极简模拟训练实现：
- CLIP/SigLIP 聚焦图文对比学习核心差异（损失函数），使用模拟图像+真实英文句子完成端到端训练；
- VLA 聚焦机器人场景下“视觉+语言指令→动作预测”的核心逻辑，模拟机器人摄像头输入+真实语言指令，训练连续动作预测能力。

## 核心差异
### CLIP vs SigLIP
| 特性                | CLIP                          | SigLIP                        |
|---------------------|-------------------------------|-------------------------------|
| 损失函数            | InfoNCE（CrossEntropy）| Sigmoid Loss（BCEWithLogits） |
| 温度系数            | 可学习参数                    | 固定值（无需学习）|
| 损失计算逻辑        | 图像/文本双视角对比损失       | 全局图文对二元交叉熵         |
| 标签构造            | 对角线为正确标签（0~B-1）| 对角线为1，其余为0（带平滑） |
| 优势                | 经典图文对比学习基线          | 训练更稳定、适合大批次场景    |

### VLA 核心特性
| 特性                | 说明                                                                 |
|---------------------|----------------------------------------------------------------------|
| 任务目标            | 输入视觉图像+语言指令，预测机器人6自由度连续动作（dx, dy, dz, roll, pitch, yaw） |
| 模型结构            | 图像编码器+文本编码器+多模态融合层+动作预测头                         |
| 损失函数            | MSE损失（连续动作回归）|
| 输入数据            | 模拟机器人摄像头图像 + CLIP官方tokenizer处理的真实语言指令            |

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

### 3. 训练VLA
```bash
python train_vla_simulated.py
```

## 代码结构说明

### 通用模块
| 模块                | 功能说明                                                                 |
|---------------------|--------------------------------------------------------------------------|
| 超参数定义          | 批次大小、学习率、特征维度等（与官方CLIP/VLA场景对齐）|
| 数据生成函数        | 生成随机模拟图像（3×224×224）+ 真实英文句子token（CLIP官方分词器处理）|
| 编码器基础结构      | 线性层实现极简编码器，聚焦核心逻辑（可替换为Transformer）|

### CLIP 核心实现
- **模型类**：`SimpleCLIP`，包含可学习温度系数
- **前向传播**：计算图像→文本、文本→图像双向相似度矩阵
- **损失函数**：`clip_loss`，双视角交叉熵损失平均（InfoNCE）

### SigLIP 核心实现
- **模型类**：`SimpleSigLIP`，温度系数固定（无需学习）
- **前向传播**：仅计算全局图文相似度矩阵（无需双向拆分）
- **损失函数**：`siglip_loss`，带标签平滑的二元交叉熵（Sigmoid Loss）

### VLA 核心实现
- **模型类**：`SimpleVLA`，包含图像编码器、文本编码器、多模态融合层、动作预测头
- **前向传播**：图像/文本特征提取→归一化→拼接融合→动作预测
- **损失函数**：`vla_loss`，MSE损失（拟合机器人6自由度连续动作）
- **推理演示**：训练完成后自动展示“语言指令→机器人动作”的预测效果

## 关键超参数说明
| 参数名             | 取值          | 说明                                                                 |
|--------------------|---------------|----------------------------------------------------------------------|
| BATCH_SIZE         | 16（CLIP/SigLIP）<br>8（VLA） | 批次大小                                                             |
| EPOCHS             | 10（CLIP/SigLIP）<br>15（VLA） | 训练轮数（CLIP/SigLIP每轮100个batch；VLA每轮50个batch）|
| LR                 | 1e-4          | 学习率（AdamW优化器）|
| TAU（SigLIP）| 0.07          | 固定温度系数（SigLIP标准取值）|
| LABEL_SMOOTHING    | 0.1           | 标签平滑系数（提升SigLIP泛化性）|
| EMBED_DIM          | 512           | 图文/VLA特征维度（与CLIP官方一致）|
| ACTION_DIM（VLA）| 6             | 机器人动作维度（dx, dy, dz, roll, pitch, yaw）|
| IMAGE_SIZE         | 224           | 输入图像尺寸（3×224×224）|
| TEXT_LEN（VLA）| 77            | CLIP固定文本token长度                                                |

## 训练输出
### 1. CLIP/SigLIP
- 训练过程实时显示每batch的损失值，每轮结束打印平均损失
- 训练完成后自动保存模型权重：
  - CLIP：`simulated_clip_real_text.pth`
  - SigLIP：`simulated_siglip_real_text.pth`

### 2. VLA
- 训练过程实时显示每batch的MSE损失值，每轮结束打印平均损失
- 训练完成后自动保存模型权重：`vla_model.pth`
- 自动执行推理演示，输出“语言指令→机器人动作”的预测结果

## 核心逻辑亮点
1. **极简实现**：用线性层替代复杂Transformer，聚焦核心业务逻辑（图文对比/VLA动作预测）
2. **数据模拟**：无需真实图像数据集，随机张量模拟图像+真实英文句子保证文本/指令合理性
3. **多任务覆盖**：同时支持图文对比学习（CLIP/SigLIP）和机器人动作预测（VLA）
4. **工程化细节**：自动检测CUDA设备、训练进度可视化（tqdm）、推理演示一键验证

## 注意事项
1. 本实现为**教学演示版**，仅保留核心逻辑，未使用CLIP/VLA官方的Transformer编码器
2. 温度系数：CLIP为可学习参数（初始值2.6592），SigLIP固定为0.07（行业标准）
3. 硬件支持：自动检测CUDA，优先使用GPU训练（CPU也可运行，速度较慢）
4. 文本处理：复用CLIP官方tokenizer，保证文本输入格式与官方一致
5. VLA动作数据：为模拟真实机器人指令对应的动作，仅作演示，可替换为真实机器人数据集

## 扩展方向
1. 替换编码器：将线性层替换为CLIP官方的Vision Transformer/Text Transformer
2. 真实数据接入：
   - CLIP/SigLIP：接入COCO、Flickr等图文数据集
   - VLA：接入真实机器人视觉+语言+动作数据集（如CALVIN、RLBench）
3. 模型优化：
   - CLIP/SigLIP：测试不同温度系数、批次大小的影响
   - VLA：添加动作正则化、多模态注意力融合、时序动作预测
4. 部署验证：将训练后的VLA模型部署到真实机器人，验证“语言指令→动作执行”效果
5. 多语言扩展：适配中文指令，替换CLIP tokenizer为多语言版本