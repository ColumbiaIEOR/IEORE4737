"""
Experiment:
    07 Compare Clean vs PGD Embeddings

Objective:
    Quantify how PGD adversarial attacks alter DINOv2 representation space
    by comparing clean and adversarial embeddings for the same CIFAR-10 images.

Inputs:
    - Paired clean/adversarial DINOv2 embeddings from Experiment 06

Outputs:
    - Mean and standard deviation of cosine similarity
    - Mean and standard deviation of Euclidean distance
    - Mean clean and adversarial embedding norms
    - Mean absolute norm change
    - Saved metrics under results/metrics/
    - PCA figure comparing clean and PGD embeddings

Research Goal:
    Determine whether successful adversarial attacks produce a measurable and
    structured shift in DINOv2 representation space.

Next Experiment:
    08_train_adversarial_detector.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from utils.results import save_metrics


INPUT_PATH = (
    "data/processed/embeddings/"
    "cifar10/pgd_100_dinov2_vits14.pt"
)

METRICS_PATH = (
    "results/metrics/"
    "07_pgd_embedding_comparison.json"
)

FIGURE_PATH = Path(
    "results/figures/"
    "07_clean_vs_pgd_pca.png"
)


def main():
    # ---------------------------------------------------------
    # 1. Load paired clean/adversarial embeddings.
    # ---------------------------------------------------------
    data = torch.load(
        INPUT_PATH,
        map_location="cpu",
    )

    clean_embeddings = (
        data["clean_embeddings"]
        .float()
    )

    adversarial_embeddings = (
        data["adversarial_embeddings"]
        .float()
    )

    labels = data["labels"]

    print(
        "Clean embeddings:",
        clean_embeddings.shape,
    )

    print(
        "Adversarial embeddings:",
        adversarial_embeddings.shape,
    )

    # ---------------------------------------------------------
    # 2. Cosine similarity.
    #
    # 1.0 means the vectors point in essentially the same direction.
    # Lower values indicate a greater representation shift.
    # ---------------------------------------------------------
    cosine_similarity = F.cosine_similarity(
        clean_embeddings,
        adversarial_embeddings,
        dim=1,
    )

    mean_cosine_similarity = (
        cosine_similarity.mean().item()
    )

    std_cosine_similarity = (
        cosine_similarity.std().item()
    )

    # ---------------------------------------------------------
    # 3. Euclidean distance.
    # ---------------------------------------------------------
    euclidean_distance = torch.norm(
        adversarial_embeddings
        - clean_embeddings,
        p=2,
        dim=1,
    )

    mean_euclidean_distance = (
        euclidean_distance.mean().item()
    )

    std_euclidean_distance = (
        euclidean_distance.std().item()
    )

    # ---------------------------------------------------------
    # 4. Embedding norms.
    # ---------------------------------------------------------
    clean_norms = torch.norm(
        clean_embeddings,
        p=2,
        dim=1,
    )

    adversarial_norms = torch.norm(
        adversarial_embeddings,
        p=2,
        dim=1,
    )

    norm_change = (
        adversarial_norms
        - clean_norms
    ).abs()

    mean_clean_norm = (
        clean_norms.mean().item()
    )

    mean_adversarial_norm = (
        adversarial_norms.mean().item()
    )

    mean_absolute_norm_change = (
        norm_change.mean().item()
    )

    # ---------------------------------------------------------
    # 5. Save numerical metrics.
    # ---------------------------------------------------------
    metrics = {
        "experiment": "07_compare_pgd_embeddings",
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "num_samples": clean_embeddings.shape[0],
        "embedding_dimension": clean_embeddings.shape[1],
        "mean_cosine_similarity": (
            mean_cosine_similarity
        ),
        "std_cosine_similarity": (
            std_cosine_similarity
        ),
        "mean_euclidean_distance": (
            mean_euclidean_distance
        ),
        "std_euclidean_distance": (
            std_euclidean_distance
        ),
        "mean_clean_embedding_norm": (
            mean_clean_norm
        ),
        "mean_adversarial_embedding_norm": (
            mean_adversarial_norm
        ),
        "mean_absolute_norm_change": (
            mean_absolute_norm_change
        ),
    }

    save_metrics(
        metrics,
        METRICS_PATH,
    )

    # ---------------------------------------------------------
    # 6. Joint PCA.
    #
    # Fit one PCA model to the combined clean/adversarial embeddings
    # so both sets are projected into the same coordinate system.
    # ---------------------------------------------------------
    combined_embeddings = torch.cat(
        [
            clean_embeddings,
            adversarial_embeddings,
        ],
        dim=0,
    ).numpy()

    pca = PCA(
        n_components=2,
    )

    reduced = pca.fit_transform(
        combined_embeddings
    )

    num_samples = clean_embeddings.shape[0]

    clean_2d = reduced[
        :num_samples
    ]

    adversarial_2d = reduced[
        num_samples:
    ]

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(9, 7),
    )

    plt.scatter(
        clean_2d[:, 0],
        clean_2d[:, 1],
        label="Clean",
        alpha=0.7,
        s=30,
    )

    plt.scatter(
        adversarial_2d[:, 0],
        adversarial_2d[:, 1],
        label="PGD",
        alpha=0.7,
        s=30,
    )

    # Draw a line between each clean/adversarial pair.
    for i in range(num_samples):
        plt.plot(
            [
                clean_2d[i, 0],
                adversarial_2d[i, 0],
            ],
            [
                clean_2d[i, 1],
                adversarial_2d[i, 1],
            ],
            alpha=0.15,
            linewidth=0.7,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        "DINOv2 Representation Shift: "
        "Clean vs PGD"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # ---------------------------------------------------------
    # 7. Print summary.
    # ---------------------------------------------------------
    print()
    print("Embedding comparison results")
    print("----------------------------")

    print(
        f"Mean cosine similarity:     "
        f"{mean_cosine_similarity:.4f}"
    )

    print(
        f"Std cosine similarity:      "
        f"{std_cosine_similarity:.4f}"
    )

    print(
        f"Mean Euclidean distance:    "
        f"{mean_euclidean_distance:.4f}"
    )

    print(
        f"Std Euclidean distance:     "
        f"{std_euclidean_distance:.4f}"
    )

    print(
        f"Mean clean norm:            "
        f"{mean_clean_norm:.4f}"
    )

    print(
        f"Mean adversarial norm:      "
        f"{mean_adversarial_norm:.4f}"
    )

    print(
        f"Mean absolute norm change:  "
        f"{mean_absolute_norm_change:.4f}"
    )

    print()
    print(
        f"Saved metrics to "
        f"{METRICS_PATH}"
    )

    print(
        f"Saved figure to "
        f"{FIGURE_PATH}"
    )


if __name__ == "__main__":
    main()