import torch


def evaluate_clip(model, dataloader, device):
    """
    CLIP 模型 图像→文本 检索精度评估函数
    评估逻辑：在一个 batch 内，第 i 张图必须匹配第 i 个文本才算正确
    精度计算方式：图搜文（image-to-text）检索准确率

    Args:
        model: 训练好的 CLIP 模型
        dataloader: 验证集/测试集 DataLoader
        device: 运行设备 cuda / cpu

    Returns:
        acc: 图像检索文本准确率（% 百分比形式）
    """
    # 切换模型为评估模式（关闭 dropout、batchnorm 等训练时才用的层）
    model.eval()

    # 初始化：正确预测数量、总样本数量
    correct = 0
    total = 0

    # 禁用梯度计算，加速推理、节省显存
    with torch.no_grad():
        # 遍历 dataloader 中的每一个 batch
        for img, txt in dataloader:
            # 将数据移动到指定设备（GPU/CPU）
            img, txt = img.to(device), txt.to(device)

            # CLIP 前向推理，得到图像-文本相似度 logits
            logits_img, _ = model(img, txt)

            # 对每个图像，找出最相似的文本索引（预测值）
            pred = logits_img.argmax(dim=1)

            # 构造标签：batch 内对角匹配（第 i 张图对应第 i 个文本）
            labels = torch.arange(len(img)).to(device)

            # 统计正确的样本数
            correct += (pred == labels).sum().item()

            # 累计总样本数
            total += len(img)

    # 计算准确率并返回（百分比形式）
    return 100 * correct / total