# ======================== 导入依赖库 ========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from clip import tokenize  # 使用真实文本分词器

# ======================== 超参数定义 ========================
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型结构参数
IMAGE_SIZE = 224
EMBED_DIM = 512
ACTION_DIM = 6    # 机器人6自由度动作：dx, dy, dz, roll, pitch, yaw
TEXT_LEN = 77     # CLIP固定文本长度

# ======================== VLA 模型定义 ========================
class SimpleVLA(nn.Module):
    def __init__(self):
        super().__init__()

        # 图像编码器：将图像转为特征向量
        self.image_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, EMBED_DIM)
        )

        # 文本编码器：将语言指令转为特征向量
        self.text_encoder = nn.Sequential(
            nn.Embedding(49408, EMBED_DIM),
            nn.Flatten(),
            nn.Linear(EMBED_DIM * TEXT_LEN, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, EMBED_DIM)
        )

        # 多模态融合层：图像特征 + 语言特征 融合
        self.fusion = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, EMBED_DIM),
            nn.ReLU()
        )

        # 动作输出层：预测机器人连续动作
        self.action_head = nn.Linear(EMBED_DIM, ACTION_DIM)

    def forward(self, image, text_tokens):
        # 1. 提取图像特征
        img_feat = self.image_encoder(image)
        img_feat = F.normalize(img_feat, dim=-1)

        # 2. 提取文本指令特征
        txt_feat = self.text_encoder(text_tokens)
        txt_feat = F.normalize(txt_feat, dim=-1)

        # 3. 拼接融合
        fused_feat = torch.cat([img_feat, txt_feat], dim=-1)
        fused_feat = self.fusion(fused_feat)

        # 4. 预测动作
        action_pred = self.action_head(fused_feat)
        return action_pred

# ======================== 真实指令 + 动作数据集 ========================
def get_real_instruction_batch(batch_size):
    """
    生成真实语言指令 + 对应机器人动作
    动作维度：dx, dy, dz, roll, pitch, yaw
    """
    # 真实机器人语言指令
    instructions = [
        "Move to the red cup and grasp it.",
        "Push the box to the right.",
        "Lift the bowl up slowly.",
        "Put the apple on the plate.",
        "Turn left and approach the chair.",
        "Stop and wait for further command.",
        "Open the gripper and release the object.",
        "Move forward a little bit.",
        "Turn right and go to the door.",
        "Pick up the pen from the table.",
        "Place the book on the shelf.",
        "Pull the drawer to open it.",
        "Move backward a small step.",
        "Rotate the bottle clockwise.",
        "Close the gripper gently.",
        "Move up to reach the object."
    ]

    # 选取batch个指令
    instructions = instructions[:batch_size]

    # 真实文本 → token
    text_tokens = tokenize(instructions).to(DEVICE)

    # 模拟图像（机器人摄像头）
    images = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    # 对应指令的真实动作标签
    action_labels = torch.tensor([
        [0.1, 0.05, 0.02, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.15, 0.0, 0.0, 0.0],
        [0.05, 0.0, 0.05, 0.0, 0.0, 0.0],
        [-0.2, 0.0, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.05, 0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.2, 0.0, 0.0, 0.0, -0.3],
        [0.08, 0.0, 0.03, 0.0, 0.0, 0.0],
        [0.02, 0.0, 0.10, 0.0, 0.0, 0.0],
        [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.12, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.20, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)[:batch_size].to(DEVICE)

    return images, text_tokens, action_labels, instructions

# ======================== VLA 训练损失 ========================
def vla_loss(pred_action, true_action):
    """
    连续动作预测使用 MSE 损失
    让模型输出的动作尽可能接近真实机器人动作
    """
    return F.mse_loss(pred_action, true_action)

# ======================== 训练主流程 ========================
def train_vla():
    # 初始化模型
    model = SimpleVLA().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"训练设备: {DEVICE}")
    print("开始训练 VLA（视觉-语言-动作）模型...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(range(50), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for _ in pbar:
            # 获取真实指令数据
            images, tokens, actions, _ = get_real_instruction_batch(BATCH_SIZE)

            # 前向传播：预测动作
            pred_actions = model(images, tokens)

            # 计算损失
            loss = vla_loss(pred_actions, actions)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / 50
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}\n")

    # 保存模型
    torch.save(model.state_dict(), "vla_model.pth")
    print("✅ VLA 模型训练完成！已保存为 vla_model.pth")

# ======================== 推理演示 ========================
def demo_inference():
    model = SimpleVLA().to(DEVICE)
    model.load_state_dict(torch.load("vla_model.pth"))
    model.eval()

    print("\n" + "="*70)
    print("           VLA 模型推理演示（语言指令 → 机器人动作）")
    print("="*70)

    images, tokens, _, instructions = get_real_instruction_batch(4)
    with torch.no_grad():
        preds = model(images, tokens)

    for i, cmd in enumerate(instructions):
        print(f"\n📝 指令: {cmd}")
        print(f"🤖 输出动作 [x y z roll pitch yaw]: {preds[i].cpu().numpy().round(3)}")

# ======================== 运行 ========================
if __name__ == "__main__":
    train_vla()   # 训练
    demo_inference()  # 推理演示