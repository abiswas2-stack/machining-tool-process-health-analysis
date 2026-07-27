"""Train and evaluate a Random Forest classifier on machining fault features."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from machining_file import MachiningFile, SUPPORTED_FEATURES


def build_feature_table(file_label_pairs, field_name="SpindleX"):
    """
    Builds a feature table across multiple files/conditions.

    Parameters:
        file_label_pairs: list of tuples (MachiningFile, label_string)
        field_name: which sensor field to use

    Returns:
        X (numpy array): rows = segments, columns = the 8 features
        y (numpy array): the label (condition name) for each row
    """
    rows = []
    labels = []

    for machining_file, label in file_label_pairs:
        n_segments = machining_file.num_segments(field_name)

        for i in range(n_segments):
            stats = machining_file.compute_stats(field_name, i)
            # Keep features in a fixed, consistent order
            feature_row = [stats[feature] for feature in SUPPORTED_FEATURES]
            rows.append(feature_row)
            labels.append(label)

    X = np.array(rows)
    y = np.array(labels)
    return X, y


def train_and_evaluate(X, y, test_size=0.3, random_state=42):
    """
    Splits data into train/test sets, trains a Random Forest, and evaluates it.

    Returns:
        A dictionary with accuracy, confusion matrix, and a text report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=sorted(set(y))),
        "labels_order": sorted(set(y)),
        "report": classification_report(y_test, predictions),
        "model": model,
    }


if __name__ == "__main__":
    baseline = MachiningFile("../data/Segmented_Machining_Baseline.mat", "BaselineCrop")
    toolwear = MachiningFile("../data/Segmented_Machining_ToolWear.mat", "ToolWearCrop")
    misalignment = MachiningFile("../data/Segmented_Machining_Misalignment.mat", "MisalignmentCrop")
    surfacecracks = MachiningFile("../data/Segmented_Machining_SurfaceCracks.mat", "SurfaceCracksCrop")

    file_label_pairs = [
        (baseline, "Baseline"),
        (toolwear, "ToolWear"),
        (misalignment, "Misalignment"),
        (surfacecracks, "SurfaceCracks"),
    ]

    X, y = build_feature_table(file_label_pairs)
    print(f"Total segments: {X.shape[0]}, Features per segment: {X.shape[1]}")

    results = train_and_evaluate(X, y)

    print(f"\nAccuracy: {results['accuracy']:.2%}")
    print(f"\nLabel order: {results['labels_order']}")
    print("Confusion matrix:")
    print(results['confusion_matrix'])
    print("\nFull report:")
    print(results['report'])