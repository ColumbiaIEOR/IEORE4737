"""
Experiment:
    09A Generate APGD Adversarial Examples

Objective:
    Generate Auto-PGD (APGD) adversarial examples against the same
    differentiable DINOv2 + linear-head CIFAR-10 classifier used for PGD.

Inputs:
    - CIFAR-10 test images in raw [0, 1] pixel space
    - Pretrained DINOv2 ViT-S/14 encoder
    - Trained CIFAR-10 linear classification head
    - Number of samples passed through --num-samples

Outputs:
    - Clean and APGD-adversarial images
    - Clean and adversarial predictions
    - Clean accuracy
    - APGD accuracy
    - Attack success rate
    - Saved attack metrics

Research Goal:
    Add a second modern gradient-based attack family and create the
    adversarial samples needed for cross-attack detector evaluation.

Next Experiment:
    09B Extract APGD DINOv2 embeddings.
"""

import argparse
from pathlib import Path

import torch
import torchattacks
from torch.utils.data import DataLoader, Subset

from detectors.linear_head import LinearHead
from encoders.dinov2 import DinoV2Encoder
from models.dinov2_classifier import DinoV2Classifier
from utils.datasets import get_cifar10_loader
from utils.results import save_metrics


MODEL_PATH = (
    "data/processed/models/"
    "dinov2_vits14_cifar10_linear_head.pt"
)

BATCH_SIZE = 20

EPSILON = 8 / 255
STEPS = 20
N_RESTARTS = 1
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of CIFAR-10 test images to attack.",
    )

    return parser.parse_args()


def get_predictions(model, images):
    with torch.no_grad():
        logits = model(images)

    return logits.argmax(dim=1)


def main():
    args = parse_args()
    num_samples = args.num_samples

    output_path = Path(
        "data/processed/adversarial/"
        f"cifar10/apgd_{num_samples}.pt"
    )

    metrics_path = (
        "results/metrics/"
        f"09a_apgd_{num_samples}_metrics.json"
    )

    # ---------------------------------------------------------
    # 1. Load CIFAR-10 in raw [0, 1] pixel space.
    # ---------------------------------------------------------
    base_loader = get_cifar10_loader(
        batch_size=BATCH_SIZE,
        train=False,
        image_size=224,
        num_workers=0,
        normalize=False,
    )

    subset = Subset(
        base_loader.dataset,
        range(num_samples),
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # 2. Load DINOv2 and trained linear head.
    # ---------------------------------------------------------
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

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # ---------------------------------------------------------
    # 3. Configure APGD.
    #
    # Same L-infinity epsilon used for PGD so the attacks are
    # directly comparable.
    # ---------------------------------------------------------
    attack = torchattacks.APGD(
        model=model,
        norm="Linf",
        eps=EPSILON,
        steps=STEPS,
        n_restarts=N_RESTARTS,
        seed=SEED,
        loss="ce",
        eot_iter=1,
        rho=0.75,
        verbose=False,
    )

    # ---------------------------------------------------------
    # 4. Generate attack.
    # ---------------------------------------------------------
    all_clean_images = []
    all_adversarial_images = []
    all_labels = []

    all_clean_predictions = []
    all_adversarial_predictions = []

    clean_correct = 0
    adversarial_correct = 0
    total = 0

    print(
        f"Generating APGD adversarial examples "
        f"for {num_samples} images..."
    )

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        clean_predictions = get_predictions(
            model,
            images,
        )

        adversarial_images = attack(
            images,
            labels,
        )

        adversarial_predictions = get_predictions(
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

        all_clean_images.append(
            images.detach().cpu()
        )

        all_adversarial_images.append(
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

        print(
            f"Processed batch "
            f"{batch_index + 1}/{len(loader)}"
        )

    # ---------------------------------------------------------
    # 5. Combine batches.
    # ---------------------------------------------------------
    clean_images = torch.cat(
        all_clean_images,
        dim=0,
    )

    adversarial_images = torch.cat(
        all_adversarial_images,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    clean_predictions = torch.cat(
        all_clean_predictions,
        dim=0,
    )

    adversarial_predictions = torch.cat(
        all_adversarial_predictions,
        dim=0,
    )

    # ---------------------------------------------------------
    # 6. Metrics.
    # ---------------------------------------------------------
    clean_accuracy = (
        clean_correct / total
    )

    adversarial_accuracy = (
        adversarial_correct / total
    )

    originally_correct_mask = (
        clean_predictions == labels
    )

    successful_attack_mask = (
        originally_correct_mask
        & (adversarial_predictions != labels)
    )

    originally_correct = (
        originally_correct_mask
        .sum()
        .item()
    )

    successful_attacks = (
        successful_attack_mask
        .sum()
        .item()
    )

    attack_success_rate = (
        successful_attacks / originally_correct
        if originally_correct > 0
        else 0.0
    )

    perturbations = (
        adversarial_images
        - clean_images
    ).abs()

    max_linf_perturbation = (
        perturbations
        .amax()
        .item()
    )

    mean_absolute_perturbation = (
        perturbations
        .mean()
        .item()
    )

    # ---------------------------------------------------------
    # 7. Save attack data.
    # ---------------------------------------------------------
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "clean_images": clean_images,
            "adversarial_images": adversarial_images,
            "labels": labels,
            "clean_predictions": clean_predictions,
            "adversarial_predictions": adversarial_predictions,
            "epsilon": EPSILON,
            "steps": STEPS,
            "n_restarts": N_RESTARTS,
            "seed": SEED,
        },
        output_path,
    )

    # ---------------------------------------------------------
    # 8. Save metrics.
    # ---------------------------------------------------------
    metrics = {
        "experiment": "09a_generate_apgd",
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "attack": "apgd",
        "num_samples": total,
        "batch_size": BATCH_SIZE,
        "epsilon": EPSILON,
        "steps": STEPS,
        "n_restarts": N_RESTARTS,
        "seed": SEED,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "originally_correct": originally_correct,
        "successful_attacks": successful_attacks,
        "attack_success_rate": attack_success_rate,
        "max_linf_perturbation": max_linf_perturbation,
        "mean_absolute_perturbation": mean_absolute_perturbation,
    }

    save_metrics(
        metrics,
        metrics_path,
    )

    # ---------------------------------------------------------
    # 9. Print summary.
    # ---------------------------------------------------------
    print()
    print("APGD results")
    print("------------")

    print(
        f"Samples:               {total}"
    )

    print(
        f"Clean accuracy:        {clean_accuracy:.4f}"
    )

    print(
        f"Adversarial accuracy:  {adversarial_accuracy:.4f}"
    )

    print(
        f"Originally correct:    {originally_correct}"
    )

    print(
        f"Successful attacks:    {successful_attacks}"
    )

    print(
        f"Attack success rate:   {attack_success_rate:.4f}"
    )

    print(
        f"Max L-inf perturbation: "
        f"{max_linf_perturbation:.6f}"
    )

    print(
        f"Configured epsilon:     "
        f"{EPSILON:.6f}"
    )

    print()
    print(
        f"Saved adversarial samples to "
        f"{output_path}"
    )

    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()