import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()

        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_dim)
        )

        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Embedding(49408, 512),
            nn.Flatten(),
            nn.Linear(512 * 77, embed_dim)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def encode_image(self, x):
        return F.normalize(self.image_encoder(x), dim=-1)

    def encode_text(self, x):
        return F.normalize(self.text_encoder(x), dim=-1)

    def forward(self, image, text):
        img_feat = self.encode_image(image)
        txt_feat = self.encode_text(text)
        logits = self.logit_scale.exp() * img_feat @ txt_feat.T
        return logits, logits.T