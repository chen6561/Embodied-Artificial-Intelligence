# 导入PyTorch深度学习框架，用于张量计算和模型构建
import torch
# 导入OpenAI的CLIP模型库，包含预训练模型和相关工具函数
import clip
# 导入PIL库的Image模块，用于图像的读取和处理
from PIL import Image

# 配置计算设备：优先使用CUDA（GPU）加速，若无则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练的CLIP模型（ViT-B/32版本）和对应的图像预处理函数
# 参数说明：
# - "ViT-B/32"：模型架构，Vision Transformer-Base，特征维度32
# - device=device：将模型加载到指定的计算设备（GPU/CPU）
model, preprocess = clip.load("ViT-B/32", device=device)

# 图像预处理与加载流程：
# 1. Image.open("CLIP.png")：读取本地的CLIP.png图像文件
# 2. preprocess()：使用CLIP模型配套的预处理函数处理图像（归一化、缩放、裁剪等）
# 3. unsqueeze(0)：给图像张量增加batch维度（从[3, 224, 224]变为[1, 3, 224, 224]）
# 4. to(device)：将图像张量移到指定计算设备（GPU/CPU）
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)

# 文本处理流程：
# 1. clip.tokenize()：将文本列表转换为CLIP模型可识别的张量格式（基于CLIP的词表编码）
#    - 输入：["a diagram", "a dog", "a cat"] 三个文本标签
#    - 输出：形状为[3, 77]的张量（77是CLIP固定的文本上下文长度）
# 2. to(device)：将文本张量移到指定计算设备（GPU/CPU）
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# 使用torch.no_grad()禁用梯度计算，节省显存并加速推理（推理阶段无需反向传播）
with torch.no_grad():
    # 提取图像特征：通过模型的encode_image方法将图像转换为特征向量
    # 输出shape：[1, 512]（batch_size=1，特征维度512）
    image_features = model.encode_image(image)
    # 提取文本特征：通过模型的encode_text方法将文本转换为特征向量
    # 输出shape：[3, 512]（3个文本，特征维度512）
    text_features = model.encode_text(text)

    # 将图像和文本特征输入模型，计算图文相似度得分
    # 返回值说明：
    # - logits_per_image：图像对每个文本的相似度得分，shape=[1, 3]
    # - logits_per_text：文本对每个图像的相似度得分，shape=[3, 1]
    logits_per_image, logits_per_text = model(image, text)

    # 计算概率分布：
    # 1. softmax(dim=-1)：对logits_per_image在最后一维做归一化，转换为概率
    # 2. cpu()：将张量从GPU移到CPU（方便后续转numpy）
    # 3. numpy()：将PyTorch张量转换为numpy数组
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 打印最终的标签概率分布，展示图像与每个文本标签的匹配概率
# 预期输出：[[0.9927937  0.00421068 0.00299572]]
# 对应含义：图像匹配"a diagram"的概率约99.28%，匹配"a dog"约0.42%，匹配"a cat"约0.30%
print("Label probs:", probs)