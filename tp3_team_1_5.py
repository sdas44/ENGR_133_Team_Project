import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib


def load_dataset(file_path, feature_cols, label_col, shuffle, seed=70):
    """
    Loads a dataset from a CSV file, separates features and labels,
    and optionally shuffles the data.
    """
    df = pd.read_csv(file_path)
    X = df[feature_cols].to_numpy()
    y = df[label_col].to_numpy()

    if shuffle:
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(y))
        X = X[shuffled_indices]
        y = y[shuffled_indices]

    return X, y


def train_val_test_split(X, y, train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    n_samples = len(X)
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    assert len(X_train) + len(X_val) + len(X_test) == n_samples
    assert len(y_train) + len(y_val) + len(y_test) == n_samples

    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_features(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=0)
    std_dev = np.std(X_train, axis=0)

    X_train_scaled = (X_train - mean) / std_dev
    X_val_scaled = (X_val - mean) / std_dev
    X_test_scaled = (X_test - mean) / std_dev

    return X_train_scaled, X_val_scaled, X_test_scaled


def calculate_metrics(predicted_labels, true_labels):
    if len(predicted_labels) != len(true_labels):
        raise ValueError("Predicted and true label arrays must have the same length.")

    correct = np.sum(predicted_labels == true_labels)
    total = len(true_labels)
    accuracy = correct / total
    error = 1.0 - accuracy

    return accuracy, error


def knn_single_prediction(new_example, X_train, y_train, k):
    distances = []
    for i in range(len(X_train)):
        distance = np.linalg.norm(new_example - X_train[i])
        distances.append((distance, y_train[i]))

    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    neighbor_labels = [label for (_, label) in k_neighbors]
    predicted_label = max(set(neighbor_labels), key=neighbor_labels.count)

    return predicted_label


def predict_labels_knn(X_new, X_train, y_train, k):
    predicted_labels = []
    for i in range(len(X_new)):
        label = knn_single_prediction(X_new[i], X_train, y_train, k)
        predicted_labels.append(label)

    return np.array(predicted_labels)


def tune_k_values(k_values, X_train, y_train, X_val, y_val, is_shuffle):
    # Initialize metrics dictionary
    metrics = {
        "acc": {"train_acc": [], "val_acc": []},
        "error": {"train_error": [], "val_error": []}
    }

    for k in k_values:
        # Predict labels for training set
        train_predictions = predict_labels_knn(X_train, X_train, y_train, k)
        train_acc, train_error = calculate_metrics(train_predictions, y_train)

        # Predict labels for validation set
        val_predictions = predict_labels_knn(X_val, X_train, y_train, k)
        val_acc, val_error = calculate_metrics(val_predictions, y_val)

        # Append metrics
        metrics["acc"]["train_acc"].append(train_acc)
        metrics["acc"]["val_acc"].append(val_acc)
        metrics["error"]["train_error"].append(train_error)
        metrics["error"]["val_error"].append(val_error)

    # Plot performance
    plot_knn_performance(k_values, metrics, is_shuffle)

    # Find best k (highest validation accuracy)
    max_val_acc = max(metrics["acc"]["val_acc"])
    best_k = k_values[metrics["acc"]["val_acc"].index(max_val_acc)]

    return best_k


def plot_knn_performance(k_values, metrics, is_shuffle):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy Plot
    axes[0].plot(k_values, metrics["acc"]["train_acc"], label="Train Accuracy")
    axes[0].plot(k_values, metrics["acc"]["val_acc"], label="Validation Accuracy")
    axes[0].set_title("Accuracy vs. k")
    axes[0].set_xlabel("k (Neighbors)")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Error Rate Plot
    axes[1].plot(k_values, metrics["error"]["train_error"], label="Train Error")
    axes[1].plot(k_values, metrics["error"]["val_error"], label="Validation Error")
    axes[1].set_title("Error Rate vs. k")
    axes[1].set_xlabel("k (Neighbors)")
    axes[1].set_ylabel("Error Rate")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    file_path = pathlib.Path(input("Enter the path to the feature dataset: ").strip())
    shuffle_choice = input("Shuffle the dataset? (yes/no): ").strip().lower()
    seed = int(input("Enter a seed for loading the dataset: ").strip())

    print(file_path)

    shuffle = (shuffle_choice == "yes")

    feature_cols = [
        'hue_mean', 'hue_std', 'saturation_mean', 'saturation_std',
        'value_mean', 'value_std', 'num_lines', 'has_circle'
    ]
    label_col = 'ClassId'

    X, y = load_dataset(file_path, feature_cols, label_col, shuffle, seed)

    # Split the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, 0.8, 0.1, 0.1)

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

    # Summary
    print("\nData loaded and split into:")
    print(f"Training set: {len(X_train_scaled)} samples")
    print(f"Validation set: {len(X_val_scaled)} samples")
    print(f"Test set: {len(X_test_scaled)} samples\n")

    # KNN tuning
    k_values = [1, 9, 20, 40, 80, 130, 200, 300, 500, 750, 1000]
    best_k = tune_k_values(k_values, X_train_scaled, y_train, X_val_scaled, y_val, shuffle)

    print(f"\nBased on the plots, the best k appears to be: {best_k}\n")

    # Final evaluation
    print(f"Evaluating final model on test set with k = {best_k}...\n")
    y_test_pred = predict_labels_knn(X_test_scaled, X_train_scaled, y_train, best_k)
    test_acc, test_error = calculate_metrics(y_test_pred, y_test)

    print("--- Final Model Performance ---")
    print(f"Test Set Accuracy: {test_acc:.4f}")
    print(f"Test Set Error Rate: {test_error:.4f}")


if __name__ == "__main__":
    main()