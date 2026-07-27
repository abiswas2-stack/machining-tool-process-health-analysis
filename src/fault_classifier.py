"""Train and evaluate a Random Forest classifier on machining fault features."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from machining_file import MachiningFile, SUPPORTED_FEATURES


def build_feature_table(file_label_pairs, field_names=("SpindleX",)):
    """
    Builds a feature table across multiple files/conditions and
    multiple sensor fields.

    Parameters:
        file_label_pairs: list of tuples (MachiningFile, label_string)
        field_names: tuple/list of sensor fields to include,
                     e.g. ("SpindleX", "PlateLFAccZ", "Power")

    Returns:
        X (numpy array): rows = segments, columns = features from
                          every field, concatenated in order
        y (numpy array): the label (condition name) for each row
        column_names (list): human-readable name for each column,
                              e.g. "SpindleX_std"
        segment_indices (numpy array): the original segment index
                                        (0, 1, 2, ...) for each row,
                                        used to test chronological splits
    """
    rows = []
    labels = []
    segment_indices = []

    for machining_file, label in file_label_pairs:
        # Use the first field to determine how many segments exist
        n_segments = machining_file.num_segments(field_names[0])

        for i in range(n_segments):
            feature_row = []
            for field_name in field_names:
                stats = machining_file.compute_stats(field_name, i)
                feature_row.extend(stats[feature] for feature in SUPPORTED_FEATURES)
            rows.append(feature_row)
            labels.append(label)
            segment_indices.append(i)

    X = np.array(rows)
    y = np.array(labels)
    segment_indices = np.array(segment_indices)

    column_names = [
        f"{field}_{feature}"
        for field in field_names
        for feature in SUPPORTED_FEATURES
    ]

    return X, y, column_names, segment_indices


def train_and_evaluate(X, y, test_size=0.3, random_state=42):
    """
    Splits data into train/test sets (randomly), trains a Random
    Forest, and evaluates it.

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


def chronological_train_test_split(X, y, segment_indices, test_fraction=0.3):
    """
    Splits data by time rather than randomly: for each condition,
    the earliest segments become the training set, and the latest
    segments become the test set. This checks whether the model
    still performs well on genuinely 'later, unseen' data from the
    same session, rather than randomly shuffled segments.

    Returns:
        X_train, X_test, y_train, y_test
    """
    train_mask = np.zeros(len(y), dtype=bool)

    for label in np.unique(y):
        label_positions = np.where(y == label)[0]
        # Sort this condition's rows by their original segment order
        ordered = label_positions[np.argsort(segment_indices[label_positions])]

        cutoff = int(len(ordered) * (1 - test_fraction))
        train_mask[ordered[:cutoff]] = True  # earliest segments -> train
        # remaining (latest segments) stay False -> test

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    return X_train, X_test, y_train, y_test


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

    sensor_fields = ("SpindleX", "PlateLFAccZ", "Power")

    X, y, column_names, segment_indices = build_feature_table(file_label_pairs, field_names=sensor_fields)
    print(f"Total segments: {X.shape[0]}, Features per segment: {X.shape[1]}")

    # --- Original random split (for comparison) ---
    results_random = train_and_evaluate(X, y)
    print(f"\n[Random split] Accuracy: {results_random['accuracy']:.2%}")
    print("Confusion matrix:")
    print(results_random['confusion_matrix'])
    print("\nFull report:")
    print(results_random['report'])

    importances = results_random['model'].feature_importances_
    top_features = sorted(zip(column_names, importances), key=lambda x: -x[1])[:5]
    print("Top 5 most important features:")
    for name, importance in top_features:
        print(f"  {name}: {importance:.4f}")

    # --- Sanity check: inspect the top suspicious features by condition ---
    df = pd.DataFrame(X, columns=column_names)
    df['label'] = y

    print("\nPower_crest_factor summary by condition:")
    print(df.groupby('label')['Power_crest_factor'].describe())

    print("\nPower_kurtosis summary by condition:")
    print(df.groupby('label')['Power_kurtosis'].describe())

    # --- Chronological split (early segments train, late segments test) ---
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        X, y, segment_indices, test_fraction=0.3
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n[Chronological split] Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=sorted(set(y))))
    print("\nFull report:")
    print(classification_report(y_test, predictions))