"""
Experiment:
    05B PGD Adversarial Sanity Check

Objective:
    Generate PGD adversarial examples against the differentiable
    DINOv2 + linear-head CIFAR-10 classifier and verify that the
    attack reduces model accuracy.

Inputs:
    - CIFAR-10 test images in raw [0, 1] pixel space
    - Pretrained DINOv2 ViT-S/14 encoder
    - Trained CIFAR-10 linear classification head

Outputs:
    - Clean accuracy on the sampled images
    - Adversarial accuracy after PGD
    - Attack success rate
    - Saved clean/adversarial image pairs and predictions

Research Goal:
    Validate the PGD attack pipeline before scaling adversarial
    generation to larger datasets.

Next Experiment:
    06_extract_pgd_embeddings.py
"""

from pathlib import Path

import torch
from torch.utils.data import Subset, DataLoader

from attacks.pgd import pgd_attack
from detectors.linear_head import LinearHead
from encoders.dinov2 import DinoV2Encoder
from models.dinov2_classifier import DinoV2Classifier
from utils.datasets import get_cifar10_loader
from utils.results import save_metrics


MODEL_PATH = (
    "data/processed/models/"
    "dinov2_vits14_cifar10_linear_head.pt"
)

OUTPUT_PATH = Path(
    "data/processed/adversarial/"
    "cifar10/pgd_100.pt"
)

METRICS_PATH = (
    "results/metrics/"
    "05b_pgd_sanity_metrics.json"
)

NUM_SAMPLES = 100
BATCH_SIZE = 20

EPSILON = 8 / 255
ALPHA = 2 / 255
STEPS = 10


def evaluate_predictions(model, images):
    with torch.no_grad():
        logits = model(images)
        return logits.argmax(dim=1)


def main():
    base_loader = get_cifar10_loader(
        batch_size=BATCH_SIZE,
        train=False,
        image_size=224,
        num_workers=0,
        normalize=False,
    )

    subset = Subset(
        base_loader.dataset,
        range(NUM_SAMPLES),
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    encoder_wrapper = DinoV2Encoder(
        model_name="dinov2_vits14",
    )

    device = encoder_wrapper.device

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    head = LinearHead(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
    )

    head.load_state_dict(
        checkpoint["model_state_dict"]
    )

    head.to(device)
    head.eval()

    model = DinoV2Classifier(
        encoder=encoder_wrapper.model,
        classifier=head,
    ).to(device)

    model.eval()

    all_clean = []
    all_adversarial = []
    all_labels = []
    all_clean_predictions = []
    all_adversarial_predictions = []

    clean_correct = 0
    adversarial_correct = 0
    total = 0

    print(
        f"Generating PGD adversarial examples "
        f"for {NUM_SAMPLES} images..."
    )

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        clean_predictions = evaluate_predictions(
            model,
            images,
        )

        adversarial_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=EPSILON,
            alpha=ALPHA,
            steps=STEPS,
            random_start=True,
        )

        adversarial_predictions = evaluate_predictions(
            model,
            adversarial_images,
        )

        clean_correct += (
            clean_predictions == labels
        ).sum().item()

        adversarial_correct += (
            adversarial_predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_clean.append(
            images.detach().cpu()
        )

        all_adversarial.append(
            adversarial_images.detach().cpu()
        )

        all_labels.append(
            labels.detach().cpu()
        )

        all_clean_predictions.append(
            clean_predictions.detach().cpu()
        )

        all_adversarial_predictions.append(
            adversarial_predictions.detach().cpu()
        )

    clean_accuracy = clean_correct / total
    adversarial_accuracy = adversarial_correct / total

    successful_attacks = sum(
        (
            clean_pred == label
            and adv_pred != label
        )
        for clean_pred, adv_pred, label in zip(
            torch.cat(all_clean_predictions),
            torch.cat(all_adversarial_predictions),
            torch.cat(all_labels),
        )
    )

    originally_correct = sum(
        (
            clean_pred == label
        )
        for clean_pred, label in zip(
            torch.cat(all_clean_predictions),
            torch.cat(all_labels),
        )
    )

    attack_success_rate = (
        successful_attacks / originally_correct
        if originally_correct > 0
        else 0.0
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "clean_images": torch.cat(all_clean),
            "adversarial_images": torch.cat(all_adversarial),
            "labels": torch.cat(all_labels),
            "clean_predictions": torch.cat(
                all_clean_predictions
            ),
            "adversarial_predictions": torch.cat(
                all_adversarial_predictions
            ),
            "epsilon": EPSILON,
            "alpha": ALPHA,
            "steps": STEPS,
        },
        OUTPUT_PATH,
    )

    metrics = {
        "experiment": "05b_generate_pgd",
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "num_samples": total,
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "steps": STEPS,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "attack_success_rate": attack_success_rate,
        "originally_correct": originally_correct,
        "successful_attacks": successful_attacks,
    }

    save_metrics(
        metrics,
        METRICS_PATH,
    )

    print()
    print(f"Clean accuracy:       {clean_accuracy:.4f}")
    print(f"Adversarial accuracy: {adversarial_accuracy:.4f}")
    print(f"Attack success rate:  {attack_success_rate:.4f}")
    print()
    print(f"Saved adversarial samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()