"""
Experiment:
    04 Linear Probe

Objective:
    Evaluate how much CIFAR-10 class information is preserved in the frozen
    DINOv2 embeddings using a simple logistic regression classifier.

Inputs:
    - Saved clean DINOv2 embeddings
    - CIFAR-10 labels

Outputs:
    - Classification accuracy
    - Precision, recall, and F1-score by class
    - Saved metrics and classification report under results/metrics/

Research Goal:
    Quantify whether the DINOv2 representations are linearly separable before
    studying how adversarial attacks alter the embedding space.

Next Experiment:
    05_generate_pgd.py
"""

import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from utils.results import save_metrics, save_text_report


EMBEDDING_PATH = (
    "data/processed/embeddings/"
    "cifar10/clean_dinov2_vits14.pt"
)

METRICS_PATH = "results/metrics/04_linear_probe_metrics.json"
REPORT_PATH = "results/metrics/04_linear_probe_classification_report.txt"

def main():
    data = torch.load(EMBEDDING_PATH)

    embeddings = data["embeddings"].numpy()
    labels = data["labels"].numpy()

    print("Embeddings:", embeddings.shape)
    print("Labels:", labels.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    print("Training linear probe...")
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )
    report = classification_report(y_test, predictions)

    print()
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print(report)

    save_metrics(
        {
            "experiment": "04_linear_probe",
            "encoder": "dinov2_vits14",
            "dataset": "cifar10",
            "embedding_dimension": embeddings.shape[1],
            "num_samples": embeddings.shape[0],
            "test_size": 0.2,
            "random_state": 42,
            "classifier": "logistic_regression",
            "accuracy": float(accuracy),
        },
        METRICS_PATH,
    )

    save_text_report(
        report,
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()