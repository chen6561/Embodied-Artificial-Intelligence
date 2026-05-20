import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from clip import tokenize


class COCOCaptionDataset(Dataset):
    """
    COCO 标题数据集加载类
    功能：加载图片 + 对应的文本描述，用于 CLIP 图像-文本检索训练
    继承自 PyTorch Dataset，必须实现 __init__, __len__, __getitem__
    """
    def __init__(self, img_dir, ann_file):
        """
        数据集初始化函数
        :param img_dir: 图片所在文件夹路径
        :param ann_file: 标注文件 JSON 路径（captions_train2017.json）
        """
        # 图片根目录
        self.img_dir = img_dir

        # 图像预处理：与 CLIP 模型输入要求一致
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),       # 统一 resize 到 224x224
            transforms.ToTensor(),                # 转为 PyTorch Tensor
            transforms.Normalize(                 # ImageNet 标准化（CLIP 通用）
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        # ===================== 读取并解析 JSON 标注文件 =====================
        # 以二进制方式读取，解决各种编码报错问题
        with open(ann_file, 'rb') as f:
            raw_data = f.read()

        # 尝试 UTF-8 解码，失败则使用 latin-1 兼容解码
        try:
            ann = json.loads(raw_data.decode('utf-8'))
        except:
            ann = json.loads(raw_data.decode('latin-1'))

        # ===================== 构建 图片ID -> 标题 的映射 =====================
        img_id_to_caption = {}
        for anno in ann['annotations']:
            img_id = anno['image_id']
            # 每个图片只保留第一个 caption（避免重复）
            if img_id not in img_id_to_caption:
                img_id_to_caption[img_id] = anno['caption']

        # ===================== 构建最终样本列表 =====================
        # samples 中每一项：(图片文件名, 标题文本)
        self.samples = []
        for img_info in ann['images']:
            img_id = img_info['id']
            filename = img_info['file_name']
            # 获取当前图片对应的 caption
            caption = img_id_to_caption.get(img_id, "")
            # 只保留有有效标题的样本
            if caption:
                self.samples.append((filename, caption))

    def __len__(self):
        """返回数据集总样本数量（必须实现）"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取单条样本（必须实现）
        :param idx: 数据索引
        :return: 预处理后的图像Tensor, 文本token, 原始标题文本
        """
        # 从样本列表中获取 文件名 和 标题
        filename, caption = self.samples[idx]
        # 拼接完整图片路径
        img_path = os.path.join(self.img_dir, filename)

        # 打开图片并转为 RGB 格式（避免灰度图/透明通道报错）
        image = Image.open(img_path).convert("RGB")
        # 图像预处理
        image = self.transform(image)

        # CLIP 文本 tokenize：将文本转为模型可接受的索引序列
        # [0] 取出 batch 维度的第一条（因为只输入了一个句子）
        tokens = tokenize([caption])[0]

        # 返回：图像张量、文本张量、原始文本（用于调试/展示）
        return image, tokens, caption


# ========================
# 数据集测试脚本
# 直接运行 dataset.py 即可测试数据集是否正常加载
# ========================
if __name__ == '__main__':
    # 测试用数据集路径（可根据自己路径修改）
    IMG_DIR = "D:/datasets/vla/coco/train2017"
    ANN_FILE = "D:/datasets/vla/coco/annotations_trainval2017/annotations/captions_train2017.json"

    # 构建数据集实例
    dataset = COCOCaptionDataset(IMG_DIR, ANN_FILE)

    # 打印数据集基本信息
    print(f"✅ 数据集加载成功！总计 {len(dataset)} 张图片")

    # 测试读取第一条数据
    img_tensor, tokens, caption = dataset[0]

    # 打印测试结果
    print("✅ 测试读取第 0 条数据成功！")
    print("🖼️  图像 shape:", img_tensor.shape)
    print("🔤 文本 tokens shape:", tokens.shape)
    print("📝 标题:", caption)