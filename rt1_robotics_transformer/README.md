# RT-1 最小化 PyTorch 复现

这是一个用于学习 RT-1 原理的简化版 PyTorch 实现。

它刻意把结构收得比较干净，方便阅读和二次修改，主要保留了 RT-1 的几部分核心思想：

- 用 `FiLM` 条件化的图像编码器处理视觉输入
- 用 `TokenLearner` 风格模块把大量视觉 token 压缩成少量 token
- 用时序 `Transformer` 聚合多帧信息
- 用离散动作 token 的分类方式预测机器人动作

它**不追求**完整复现原论文/原仓库中的工程细节，因此没有包含：

- 原版的 `EfficientNet` 图像 tokenizer
- `Universal Sentence Encoder` 文本编码流程
- 真正的机器人轨迹数据读取
- 大规模分布式训练与部署基础设施

## 文件说明

- `rt1_model.py`：模型结构与动作离散化工具函数
- `train_demo.py`：一个可运行的合成数据训练示例

## RT-1 的核心流程

```text
多帧图像 + 文本指令
    -> 带语言条件的视觉编码器
    -> TokenLearner 压缩视觉 token
    -> Transformer 建模时间上下文
    -> 输出离散动作 token
```

## 快速运行

在当前目录执行：

```bash
python train_demo.py
```

如果环境里已经安装了 `torch`，你会看到：

- 每隔 10 个 step 打印一次 loss
- 训练结束后打印一个样例的预测结果

## 张量形状

- 图像输入：`[batch, time, 3, H, W]`
- 文本嵌入：`[batch, text_dim]`
- 每帧压缩后的 token 数：`tokens_per_frame`
- Transformer 输入：`[batch, time * tokens_per_frame, embed_dim]`
- 输出 logits：`[batch, num_action_dims, vocab_size]`

## 这个版本和论文的对应关系

这个教学版保留了 RT-1 的高层思路：

1. 把每一帧图像转成一组视觉 token。
2. 用语言信息调制视觉特征提取。
3. 用 token 压缩降低计算量。
4. 用 Transformer 处理多帧时间上下文。
5. 把动作离散化为分类任务来学习。

## 如果你想继续往“更像论文复现”推进

下一步可以逐步增强为：

- 把简化 CNN 换成更强的视觉 backbone
- 把随机文本嵌入换成真实文本编码器
- 把合成数据集换成真实轨迹数据
- 加入因果 mask 和更接近原论文的自回归动作预测

## 当前限制

这个工作区里我已经确认代码语法没问题，但当前 Python 环境还没有安装 `torch`，所以如果直接运行会报缺少依赖。安装 `torch` 后即可运行演示脚本。
