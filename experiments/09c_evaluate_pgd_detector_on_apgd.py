"""
Experiment:
    09C PGD-to-APGD Cross-Attack Evaluation

Objective:
    Evaluate whether the representation-space detector trained ONLY on
    PGD embeddings generalizes to APGD adversarial examples.

Inputs:
    - Frozen PGD-trained logistic-regression detector from Experiment 08
    - Clean/APGD DINOv2 embeddings from Experiment 09B

Method:
    - Do not retrain or refit the detector.
    - Construct a balanced clean/APGD dataset.
    - Apply the existing PGD detector directly.
    - Measure accuracy, precision, recall, F1, and ROC-AUC.

Outputs:
    - Cross-attack detection metrics
    - Confusion matrix
    - Saved metrics JSON

Research Goal:
    Determine whether the PGD detector has learned a PGD-specific feature
    signature or a representation-space pattern that transfers to a
    different adversarial attack.

Important:
    The detector and its StandardScaler remain completely frozen.
"""

import argparse
import pickle

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from utils.results import save_metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of clean/APGD image pairs to evaluate.",
    )

    parser.add_argument(
        "--detector-samples",
        type=int,
        default=1000,
        help="Sample count used when training the saved PGD detector.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    num_samples = args.num_samples
    detector_samples = args.detector_samples

    embedding_path = (
        "data/processed/embeddings/"
        f"cifar10/apgd_{num_samples}_dinov2_vits14.pt"
    )

    detector_path = (
        "data/processed/models/"
        f"pgd_detector_{detector_samples}_logistic.pkl"
    )

    metrics_path = (
        "results/metrics/"
        f"09c_pgd_detector_on_apgd_{num_samples}.json"
    )

    # ---------------------------------------------------------
    # 1. Load APGD embeddings.
    # ---------------------------------------------------------
    data = torch.load(
        embedding_path,
        map_location="cpu",
    )

    clean_embeddings = (
        data["clean_embeddings"]
        .float()
        .numpy()
    )

    apgd_embeddings = (
        data["adversarial_embeddings"]
        .float()
        .numpy()
    )

    if clean_embeddings.shape[0] != num_samples:
        raise ValueError(
            f"Expected {num_samples} image pairs, "
            f"found {clean_embeddings.shape[0]}."
        )

    # ---------------------------------------------------------
    # 2. Construct balanced evaluation set.
    #
    # 0 = clean
    # 1 = adversarial
    # ---------------------------------------------------------
    X_test = np.concatenate(
        [
            clean_embeddings,
            apgd_embeddings,
        ],
        axis=0,
    )

    y_test = np.concatenate(
        [
            np.zeros(
                num_samples,
                dtype=np.int64,
            ),
            np.ones(
                num_samples,
                dtype=np.int64,
            ),
        ],
        axis=0,
    )

    # ---------------------------------------------------------
    # 3. Load FROZEN PGD detector.
    #
    # No fit(), refit(), or training occurs here.
    # ---------------------------------------------------------
    with open(
        detector_path,
        "rb",
    ) as file:
        detector = pickle.load(file)

    print(
        "Loaded frozen PGD detector:"
    )

    print(
        detector_path
    )

    print()
    print(
        f"Evaluating on "
        f"{num_samples} clean + "
        f"{num_samples} APGD embeddings..."
    )

    # ---------------------------------------------------------
    # 4. Evaluate.
    # ---------------------------------------------------------
    predictions = detector.predict(
        X_test
    )

    probabilities = detector.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "clean",
            "APGD",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        predictions,
        target_names=[
            "clean",
            "APGD",
        ],
        zero_division=0,
    )

    # ---------------------------------------------------------
    # 5. Save metrics.
    # ---------------------------------------------------------
    metrics = {
        "experiment": (
            "09c_evaluate_pgd_detector_on_apgd"
        ),
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "detector_training_attack": "pgd",
        "evaluation_attack": "apgd",
        "num_image_pairs": num_samples,
        "num_embeddings": int(
            len(X_test)
        ),
        "accuracy": float(
            accuracy
        ),
        "roc_auc": float(
            roc_auc
        ),
        "clean_precision": float(
            report["clean"]["precision"]
        ),
        "clean_recall": float(
            report["clean"]["recall"]
        ),
        "clean_f1": float(
            report["clean"]["f1-score"]
        ),
        "apgd_precision": float(
            report["APGD"]["precision"]
        ),
        "apgd_recall": float(
            report["APGD"]["recall"]
        ),
        "apgd_f1": float(
            report["APGD"]["f1-score"]
        ),
        "confusion_matrix": (
            cm.tolist()
        ),
    }

    save_metrics(
        metrics,
        metrics_path,
    )

    # ---------------------------------------------------------
    # 6. Print summary.
    # ---------------------------------------------------------
    print()
    print("PGD -> APGD cross-attack results")
    print("--------------------------------")

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print()
    print("Confusion matrix")
    print("----------------")

    print(cm)

    print()
    print(report_text)

    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()