# RT-2 教学版 PyTorch 复现

这是一份面向教学和原理理解的 **RT-2** 简化复现代码。

它不是对 Google 官方工程的逐行照搬，而是保留 RT-2 最关键的算法思路，并用更小、更容易读懂的 PyTorch 代码表达出来。

## 这份代码保留了什么

1. **统一词表**
   文本 token 和机器人动作 token 放在同一个输出空间里。

2. **统一生成**
   同一个模型既能回答 VQA 风格问题，也能输出机器人动作 token。

3. **动作离散化**
   连续控制量先离散，再作为 token 生成，这对应 RT-1 / RT-2 的核心做法。

4. **混合训练**
   数据集同时包含“问答样本”和“机器人样本”，体现 RT-2 的 co-fine-tuning 思路。

5. **受约束动作解码**
   生成机器人动作时，模型每一步只允许输出当前动作维度对应的 token。

## 这份代码做了哪些简化

1. 没有复现原始的大型 VLM 底座（例如 PaLI-X / PaLM-E）。
2. 没有接真实机器人数据集，而是构造了一个可控的合成几何场景。
3. 没有复现真实部署、云端 TPU、超大规模训练等工程部分。
4. 没有做论文里的全部 benchmark，只做最小可运行训练与推理演示。

## 文件说明

- `rt2_tokenizer.py`
  统一 tokenizer，负责：
  - 文本编码/解码
  - 动作离散化/反离散化
  - 动作 token 约束列表生成

- `synthetic_data.py`
  合成一个“小世界”数据集：
  - VQA 任务：例如“最左边是什么颜色”
  - Robot 任务：例如“抓红色目标”

- `rt2_model.py`
  教学版 RT-2 主模型：
  - 多帧图像编码为视觉前缀 token
  - 文本和动作共享统一 token 空间
  - Transformer 做 prefix-LM 风格建模

- `train_demo.py`
  一个最小训练脚本，训练结束后会分别演示：
  - 文本回答
  - 动作 token 生成

## 算法思路对应

可以把这份代码理解成下面这个流程：

```text
多帧图像 + 文本指令
    -> 视觉编码器提取 image tokens
    -> 文本 token 作为语言前缀/目标
    -> Transformer 统一建模
    -> 输出可以是普通文本，也可以是动作 token
```

这正是 RT-2 的关键思想：

```text
Vision-Language Model + Action Tokenization = Vision-Language-Action Model
```

## 运行环境

目标环境：

- Ubuntu 22.04
- Python 3.10+ 或 3.11+
- PyTorch 2.x

最小依赖：

```bash
pip install torch
```

运行方式：

```bash
cd work/rt2_tutorial
python train_demo.py
```

## Win10 + VSCode 调试

这个工程目录里附带了一个 `.vscode/launch.json`，直接用 VSCode 打开 `work/rt2_tutorial` 后即可点运行调试。

如果你在 Windows 上调试、在 Ubuntu 上运行，建议保持：

1. 两边 Python 主版本一致
2. 两边都使用 UTF-8
3. 两边都安装同版本的 `torch`

## 进一步扩展建议

如果你想把这份教学版往“更像论文复现”方向推进，可以按下面顺序升级：

1. 用更强的视觉 backbone 替换小型 CNN
2. 把合成 VQA 数据换成真实图文问答数据
3. 把合成动作任务换成真实机器人 trajectory
4. 加入显式的 planning / chain-of-thought 文本前缀
5. 做更完整的动作受约束生成和多任务评测

## 当前限制

在当前这个 Codex 工作区里，我可以做语法检查，但如果本机 Python 没装 `torch`，就不能直接把训练跑起来。代码本身是按 Ubuntu 22.04 + PyTorch 环境编写的。
