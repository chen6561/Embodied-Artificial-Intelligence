import torch

def evaluate_clip(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, txt in dataloader:
            img, txt = img.to(device), txt.to(device)
            logits_img, _ = model(img, txt)
            pred = logits_img.argmax(dim=1)
            labels = torch.arange(len(img)).to(device)
            correct += (pred == labels).sum().item()
            total += len(img)
    return 100 * correct / total