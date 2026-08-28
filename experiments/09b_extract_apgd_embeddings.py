"""
Experiment:
    09B Extract APGD Embeddings

Objective:
    Extract frozen DINOv2 representations for the clean and APGD
    adversarial CIFAR-10 image pairs generated in Experiment 09A.

Inputs:
    - Saved clean/APGD image pairs
    - Pretrained DINOv2 ViT-S/14 encoder
    - Number of samples passed through --num-samples

Outputs:
    - Clean DINOv2 embeddings
    - APGD DINOv2 embeddings
    - Labels and predictions
    - Saved paired embedding file

Research Goal:
    Produce APGD representation data for cross-attack evaluation
    against the detector trained only on PGD.

Next Experiment:
    09C Evaluate frozen PGD detector on APGD.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from encoders.dinov2 import DinoV2Encoder


BATCH_SIZE = 20


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of APGD image pairs to encode.",
    )

    return parser.parse_args()


def extract_embeddings(
    encoder,
    images,
    batch_size,
):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=images.dtype,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=images.dtype,
    ).view(1, 3, 1, 1)

    dataset = TensorDataset(images)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_embeddings = []

    for (batch_images,) in tqdm(
        loader,
        desc="Extracting embeddings",
    ):
        normalized_images = (
            batch_images - mean
        ) / std

        embeddings = encoder.encode(
            normalized_images
        )

        all_embeddings.append(
            embeddings
        )

    return torch.cat(
        all_embeddings,
        dim=0,
    )


def main():
    args = parse_args()
    num_samples = args.num_samples

    input_path = (
        "data/processed/adversarial/"
        f"cifar10/apgd_{num_samples}.pt"
    )

    output_path = Path(
        "data/processed/embeddings/"
        f"cifar10/apgd_{num_samples}_dinov2_vits14.pt"
    )

    data = torch.load(
        input_path,
        map_location="cpu",
    )

    clean_images = (
        data["clean_images"]
        .float()
    )

    adversarial_images = (
        data["adversarial_images"]
        .float()
    )

    labels = (
        data["labels"]
        .long()
    )

    clean_predictions = (
        data["clean_predictions"]
        .long()
    )

    adversarial_predictions = (
        data["adversarial_predictions"]
        .long()
    )

    if clean_images.shape[0] != num_samples:
        raise ValueError(
            f"Requested {num_samples} samples, "
            f"but input contains {clean_images.shape[0]}."
        )

    print(
        "Clean images:",
        clean_images.shape,
    )

    print(
        "APGD images:",
        adversarial_images.shape,
    )

    encoder = DinoV2Encoder(
        model_name="dinov2_vits14",
    )

    print()
    print("Encoding clean images...")

    clean_embeddings = extract_embeddings(
        encoder,
        clean_images,
        BATCH_SIZE,
    )

    print()
    print("Encoding APGD images...")

    adversarial_embeddings = extract_embeddings(
        encoder,
        adversarial_images,
        BATCH_SIZE,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "clean_embeddings": clean_embeddings,
            "adversarial_embeddings": adversarial_embeddings,
            "labels": labels,
            "clean_predictions": clean_predictions,
            "adversarial_predictions": adversarial_predictions,
            "epsilon": data["epsilon"],
            "steps": data["steps"],
        },
        output_path,
    )

    print()
    print(
        "Clean embeddings:",
        clean_embeddings.shape,
    )

    print(
        "APGD embeddings:",
        adversarial_embeddings.shape,
    )

    print()
    print(
        f"Saved paired embeddings to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()