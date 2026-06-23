from dataclasses import asdict

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from rt1_model import RT1Config, RT1Model, discretize_action, undiscretize_action


class ToyRT1Dataset(Dataset):
    """
    Synthetic data that makes the learning problem non-trivial but still solvable.
    The action tokens are deterministic functions of the visual sequence and text.
    """

    def __init__(self, config: RT1Config, num_samples: int = 256) -> None:
        self.config = config
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        g = torch.Generator().manual_seed(index)
        images = torch.rand(
            self.config.sequence_length,
            3,
            self.config.image_size,
            self.config.image_size,
            generator=g,
        )
        text = torch.randn(self.config.text_dim, generator=g)

        # Build an action target from frame statistics so the model has a signal to learn.
        frame_stats = images.mean(dim=(2, 3))
        temporal_stats = frame_stats.mean(dim=0)
        text_stats = text.view(self.config.num_action_dims, -1).mean(dim=1)
        continuous_action = torch.tanh(
            0.7 * temporal_stats[: self.config.num_action_dims] + 0.3 * text_stats
        )
        action_tokens = discretize_action(continuous_action, self.config.vocab_size)
        return images, text, action_tokens


def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def main() -> None:
    config = RT1Config()
    print("Config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    dataset = ToyRT1Dataset(config, num_samples=384)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RT1Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    print(f"Training on: {device}")
    model.train()
    for epoch in range(3):
        running_loss = 0.0
        for step, (images, text, target_tokens) in enumerate(loader, start=1):
            images = images.to(device)
            text = text.to(device)
            target_tokens = target_tokens.to(device)

            logits = model(images, text)
            loss = compute_loss(logits, target_tokens)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 10 == 0:
                avg_loss = running_loss / 10
                print(f"epoch={epoch + 1} step={step:03d} loss={avg_loss:.4f}")
                running_loss = 0.0

    model.eval()
    with torch.no_grad():
        images, text, target_tokens = dataset[0]
        logits = model(images.unsqueeze(0).to(device), text.unsqueeze(0).to(device))
        predicted_tokens = logits.argmax(dim=-1).squeeze(0).cpu()
        predicted_action = undiscretize_action(predicted_tokens, config.vocab_size)
        target_action = undiscretize_action(target_tokens, config.vocab_size)

    print("\nSample prediction")
    print("target tokens   :", target_tokens.tolist())
    print("predicted tokens:", predicted_tokens.tolist())
    print("target action   :", [round(x, 3) for x in target_action.tolist()])
    print("predicted action:", [round(x, 3) for x in predicted_action.tolist()])


if __name__ == "__main__":
    main()
