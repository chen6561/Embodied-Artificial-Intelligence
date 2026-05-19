# 导入必要的依赖库
import numpy as np
import pytest
import torch
from PIL import Image

# 导入CLIP模型库（OpenAI的Contrastive Language-Image Pre-training）
import clip


# 使用pytest的参数化装饰器，遍历CLIP所有可用的模型名称
# 目的：对每一个CLIP模型都执行一次一致性测试，确保所有模型都符合预期
@pytest.mark.parametrize('model_name', clip.available_models())
def test_consistency(model_name):
    """
    测试CLIP模型的JIT编译版本与纯Python版本的输出一致性
    核心逻辑：验证同一输入下，JIT模型和Python模型的输出概率分布是否近似相等

    Args:
        model_name (str): 待测试的CLIP模型名称（由clip.available_models()生成）
    """
    # 设置计算设备为CPU（避免GPU环境依赖，保证测试可复现）
    device = "cpu"

    # 加载JIT编译版本的CLIP模型和对应的图像预处理变换
    # jit=True：加载TorchScript编译后的模型（优化推理性能）
    jit_model, transform = clip.load(model_name, device=device, jit=True)

    # 加载纯Python版本的CLIP模型（不使用JIT编译）
    # 忽略返回的transform（与JIT版本相同，无需重复获取）
    py_model, _ = clip.load(model_name, device=device, jit=False)

    # 加载测试图像并执行预处理：
    # 1. 打开CLIP.png图像文件
    # 2. 应用模型要求的预处理变换（归一化、尺寸调整等）
    # 3. 添加batch维度（模型输入要求批量数据，shape: [1, 3, H, W]）
    # 4. 将张量移至指定设备（CPU）
    image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)

    # 构建测试文本并进行tokenize处理：
    # 1. 定义测试文本列表（包含3个不同文本）
    # 2. 使用CLIP的tokenizer将文本转换为模型可接受的张量格式
    # 3. 将张量移至指定设备（CPU）
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    # 禁用梯度计算（推理阶段无需反向传播，提升速度并节省内存）
    with torch.no_grad():
        # 用JIT模型执行前向推理，获取图像-文本匹配的logits
        # 返回值：logits_per_image（图像对各文本的匹配分数）, logits_per_text（文本对各图像的匹配分数）
        # 此处仅关注logits_per_image，忽略logits_per_text（用_接收）
        logits_per_image, _ = jit_model(image, text)
        # 将logits转换为概率分布（softmax按最后一维计算）
        # 转换为CPU上的numpy数组（方便后续数值比较）
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # 用纯Python模型执行相同的前向推理流程，获取概率分布
        logits_per_image, _ = py_model(image, text)
        py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # 断言JIT模型和Python模型的输出概率分布近似相等
    # np.allclose：验证两个数组的元素是否在指定容差范围内相等
    # atol=0.01：绝对误差容限（允许元素间绝对差值≤0.01）
    # rtol=0.1：相对误差容限（允许元素间相对差值≤10%）
    # 目的：允许数值计算的微小误差（浮点精度、JIT编译优化等），同时保证核心结果一致
    assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)