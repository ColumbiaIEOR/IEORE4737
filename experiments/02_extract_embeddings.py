"""
Experiment:
    02 DINOv2 Embedding Extraction

Objective:
    Extract frozen DINOv2 feature representations for CIFAR-10 images using
    the pretrained DINOv2 ViT-S/14 encoder.

Inputs:
    - CIFAR-10 test dataset
    - Pretrained DINOv2 ViT-S/14 encoder

Outputs:
    - 384-dimensional embedding for each image
    - CIFAR-10 labels
    - Saved embedding file under data/processed/embeddings/

Research Goal:
    Build the clean visual representation baseline that will later be compared
    against adversarially perturbed examples.

Next Experiment:
    03_pca_clean.py
"""

from encoders.dinov2 import DinoV2Encoder
from pipelines.dinov2_pipeline import (
    extract_embeddings,
    save_embeddings,
)
from utils.datasets import get_cifar10_loader


def main():
    loader = get_cifar10_loader(
        batch_size=64,
        train=False,
        image_size=224,
        num_workers=0,
    )

    encoder = DinoV2Encoder(
        model_name="dinov2_vits14",
    )

    embeddings, labels = extract_embeddings(
        encoder=encoder,
        loader=loader,
    )

    print("Embeddings:", embeddings.shape)
    print("Labels:", labels.shape)

    save_embeddings(
        embeddings=embeddings,
        labels=labels,
        output_path=(
            "data/processed/embeddings/"
            "cifar10/clean_dinov2_vits14.pt"
        ),
    )


if __name__ == "__main__":
    main()