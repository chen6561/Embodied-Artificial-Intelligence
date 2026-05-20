import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from clip import tokenize

class COCOCaptionDataset(Dataset):
    def __init__(self, img_dir, ann_file):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # 读取 JSON（解决所有编码问题）
        with open(ann_file, 'rb') as f:
            raw_data = f.read()
        try:
            ann = json.loads(raw_data.decode('utf-8'))
        except:
            ann = json.loads(raw_data.decode('latin-1'))

        # 建立图片ID -> 标题
        img_id_to_caption = {}
        for anno in ann['annotations']:
            img_id = anno['image_id']
            if img_id not in img_id_to_caption:
                img_id_to_caption[img_id] = anno['caption']

        # 构建数据列表
        self.samples = []
        for img_info in ann['images']:
            img_id = img_info['id']
            filename = img_info['file_name']
            caption = img_id_to_caption.get(img_id, "")
            if caption:
                self.samples.append((filename, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = tokenize([caption])[0]
        return image, tokens, caption


# ========================
# 这里是 MAIN 测试函数
# 直接运行 dataset.py 就会自动测试
# ========================
if __name__ == '__main__':
    # 你的路径（已经正确）
    IMG_DIR = "D:/datasets/vla/coco/train2017"
    ANN_FILE = "D:/datasets/vla/coco/annotations_trainval2017/annotations/captions_train2017.json"

    # 加载数据集
    dataset = COCOCaptionDataset(IMG_DIR, ANN_FILE)

    # 输出数据集信息
    print(f"✅ 数据集加载成功！总计 {len(dataset)} 张图片")

    # 测试读取第 0 条数据
    img_tensor, tokens, caption = dataset[0]

    print("✅ 测试读取第 0 条数据成功！")
    print("🖼️  图像 shape:", img_tensor.shape)
    print("🔤 文本 tokens shape:", tokens.shape)
    print("📝 标题:", caption)