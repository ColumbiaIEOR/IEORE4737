"""
Experiment:
    05A Train Differentiable Linear Head

Objective:
    Train a lightweight PyTorch classifier on the previously extracted
    DINOv2 CIFAR-10 embeddings.

Inputs:
    - Clean DINOv2 embeddings
    - CIFAR-10 labels

Outputs:
    - Trained linear head weights
    - Training and test accuracy

Research Goal:
    Create a differentiable DINOv2 classification pipeline that can be
    attacked end-to-end using gradient-based adversarial methods.

Next Experiment:
    05b_generate_pgd.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from detectors.linear_head import LinearHead


EMBEDDING_PATH = (
    "data/processed/embeddings/"
    "cifar10/clean_dinov2_vits14.pt"
)

OUTPUT_PATH = Path(
    "data/processed/models/"
    "dinov2_vits14_cifar10_linear_head.pt"
)

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = model(embeddings)
            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


def main():
    torch.manual_seed(SEED)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    data = torch.load(
        EMBEDDING_PATH,
        map_location="cpu",
    )

    embeddings = data["embeddings"].float()
    labels = data["labels"].long()

    dataset = TensorDataset(
        embeddings,
        labels,
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = LinearHead(
        input_dim=384,
        num_classes=10,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"Training on {device}...")

    best_accuracy = 0.0
    best_state_dict = None
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            logits = model(batch_embeddings)

            loss = criterion(
                logits,
                batch_labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        accuracy = evaluate(
            model,
            test_loader,
            device,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1

            best_state_dict = {
                key: value.detach().cpu().clone()   
                for key, value in model.state_dict().items()
            }
        
        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
            or epoch == EPOCHS - 1
        ):
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} "
                f"| Loss: {running_loss / len(train_loader):.4f} "
                f"| Test Accuracy: {accuracy:.4f}"
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "input_dim": 384,
            "num_classes": 10,
            "test_accuracy": best_accuracy,
            "best_epoch": best_epoch,
        },
        OUTPUT_PATH,
    )

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Saved model to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()