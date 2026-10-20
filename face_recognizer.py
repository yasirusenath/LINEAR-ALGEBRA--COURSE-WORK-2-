"""Face recognition wrapper for CW2 Tasks 2.1, 2.2, and 2.3.

Expected dataset structure:
    images/
        student_1/  image files...
        student_2/  image files...
        student_3/  image files...

Task 2.1: load grayscale images, resize to 300x300, flatten to 90,000 values,
          and split the data into stratified train/test sets.
Task 2.2: fit PCA using the training split only, transform train/test data,
          and visualize eigenfaces.
Task 2.3: train Logistic Regression and evaluate the classifier.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Keep this label mapping fixed throughout the notebook and project.
CLASS_TO_LABEL = {
    "student_1": 0,
    "student_2": 1,
    "student_3": 2,
}

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif"
}


class FaceRecognizerWrapper:
    """Wrapper for face image loading, splitting, PCA/Eigenfaces and classification."""

    def __init__(self, img_size=300, n_components=50, seed=42):
        if not isinstance(img_size, int) or img_size <= 0:
            raise ValueError("img_size must be a positive integer.")
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer.")

        self.img_size = img_size
        self.n_components = n_components
        self.seed = seed
        self.class_names = list(CLASS_TO_LABEL.keys())

        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.pca = None

        self.X = None
        self.y = None
        self.image_paths = []

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_train_pca = None
        self.X_test_pca = None

    def load_and_preprocess_data(self, DATASET_DIR):
        """Load each class folder as grayscale, resize, flatten and scale images."""
        dataset_path = Path(DATASET_DIR)
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

        missing_folders = [
            name for name in self.class_names
            if not (dataset_path / name).is_dir()
        ]
        if missing_folders:
            raise FileNotFoundError(
                f"Missing class folder(s) in {dataset_path}: {missing_folders}. "
                f"Expected folders: {self.class_names}"
            )

        data_list = []
        labels_list = []
        path_list = []

        for class_name, label in CLASS_TO_LABEL.items():
            class_folder = dataset_path / class_name
            class_image_count = 0

            # rglob also supports images stored in subfolders inside a class folder.
            for img_path in sorted(class_folder.rglob("*")):
                if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: OpenCV could not read image: {img_path}")
                    continue

                image = cv2.resize(image, (self.img_size, self.img_size))
                flattened = image.reshape(-1)

                data_list.append(flattened)
                labels_list.append(label)
                path_list.append(str(img_path))
                class_image_count += 1

            print(f"Class '{class_name}' (label {label}): {class_image_count} images loaded")

        if not data_list:
            raise ValueError(
                f"No supported images were loaded from {dataset_path}. "
                "Check the class folders and image files."
            )

        self.X = np.asarray(data_list, dtype=np.float32) / 255.0
        self.y = np.asarray(labels_list, dtype=np.int64)
        self.image_paths = path_list

        class_counts = {
            name: int(np.sum(self.y == label))
            for name, label in CLASS_TO_LABEL.items()
        }
        too_small = {name: count for name, count in class_counts.items() if count < 2}
        if too_small:
            raise ValueError(
                "Each class needs at least two readable images for a stratified split. "
                f"Current counts: {class_counts}"
            )

        print("\nDataset loaded successfully.")
        print("Fixed label mapping:", CLASS_TO_LABEL)
        print("Total images:", len(self.X))
        print("Feature matrix shape:", self.X.shape)
        print(f"Each image has {self.img_size} x {self.img_size} = {self.img_size ** 2} features.")

    def split_data(self, train_size=0.80, test_size=0.20):
        """Create stratified train/test splits using the configured random seed."""
        if self.X is None or self.y is None:
            raise ValueError("Load the dataset before splitting.")
        if not np.isclose(train_size + test_size, 1.0):
            raise ValueError("train_size and test_size must add up to 1.0.")
        if not (0 < train_size < 1 and 0 < test_size < 1):
            raise ValueError("Both train_size and test_size must be greater than 0 and less than 1.")

        class_counts = np.array([np.sum(self.y == label) for label in CLASS_TO_LABEL.values()])
        n_classes = len(CLASS_TO_LABEL)
        n_samples = len(self.y)
        n_train = int(np.floor(n_samples * train_size))
        n_test = n_samples - n_train

        if np.min(class_counts) < 2 or n_train < n_classes or n_test < n_classes:
            raise ValueError(
                "Not enough images to perform a stratified train/test split. "
                f"Total={n_samples}, class counts={class_counts.tolist()}, "
                f"train samples={n_train}, test samples={n_test}."
            )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.X,
            self.y,
            train_size=train_size,
            test_size=test_size,
            random_state=self.seed,
            stratify=self.y,
        )

        # A new split invalidates any previously fitted PCA features/model.
        self.pca = None
        self.X_train_pca = None
        self.X_test_pca = None
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)

        print(f"Training proportion: {train_size * 100:.0f}%")
        print(f"Testing proportion: {test_size * 100:.0f}%")
        print("Training images:", len(self.X_train))
        print("Testing images:", len(self.X_test))

    def fit_pca(self, n_components=None):
        """Fit PCA on training data only and transform train and test data."""
        if self.X_train is None or self.X_test is None:
            raise ValueError("Run split_data() or create X_train/X_test before fitting PCA.")

        if n_components is None:
            n_components = self.n_components
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer (for this task, use 50).")

        # Randomized SVD is much more practical for 90,000-pixel images and still
        # computes an approximate PCA/SVD solution for the requested leading components.
        max_allowed = min(self.X_train.shape) - 1
        if n_components > max_allowed:
            raise ValueError(
                f"Cannot calculate {n_components} components with training matrix shape "
                f"{self.X_train.shape}. Reduce n_components or add more training images. "
                f"For the randomized SVD solver, use at most {max_allowed} components."
            )

        self.n_components = n_components
        self.pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=self.seed,
        )

        # Important: fit only on X_train to prevent test-set leakage.
        self.X_train_pca = self.pca.fit_transform(self.X_train)
        self.X_test_pca = self.pca.transform(self.X_test)

        # PCA changed the feature space, so any old classifier fit is no longer current.
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)

        print("PCA components:", self.pca.n_components_)
        print("Training features before PCA:", self.X_train.shape)
        print("Testing features before PCA:", self.X_test.shape)
        print("Training features after PCA:", self.X_train_pca.shape)
        print("Testing features after PCA:", self.X_test_pca.shape)
        print("Explained variance by selected components:", f"{self.pca.explained_variance_ratio_.sum() * 100:.2f}%")

        return self.X_train_pca, self.X_test_pca

    def plot_eigenfaces(self, n_components=5):
        """Plot the first n_components PCA components as grayscale eigenfaces."""
        if self.pca is None:
            raise ValueError("Call fit_pca() before plotting eigenfaces.")
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components to display must be a positive integer.")
        if n_components > len(self.pca.components_):
            raise ValueError(
                f"You can display at most {len(self.pca.components_)} fitted components."
            )

        n_cols = min(5, n_components)
        n_rows = int(np.ceil(n_components / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        axes = np.array(axes, dtype=object).reshape(-1)

        for index in range(n_components):
            eigenface = self.pca.components_[index].reshape(self.img_size, self.img_size)
            axes[index].imshow(eigenface, cmap="gray")
            axes[index].set_title(f"Eigenface {index + 1}")
            axes[index].axis("off")

        for index in range(n_components, len(axes)):
            axes[index].axis("off")

        fig.suptitle("First Principal Components (Eigenfaces)", fontsize=14)
        plt.tight_layout()
        plt.show()

    def train_classifier(self):
        """Train the multi-class Logistic Regression model on PCA features."""
        if self.X_train_pca is None or self.y_train is None:
            raise ValueError("Run fit_pca() before training the classifier.")

        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.model.fit(self.X_train_pca, self.y_train)
        print("Logistic Regression training completed successfully.")

    def apply_pca_and_train(self):
        """Backward-compatible helper that performs Task 2.2 and then Task 2.3."""
        self.fit_pca(self.n_components)
        self.train_classifier()

    def evaluate(self):
        """Evaluate the fitted classifier on the held-out test data."""
        if self.X_test_pca is None:
            raise ValueError("Run fit_pca() before evaluating the model.")
        if not hasattr(self.model, "classes_"):
            raise ValueError("Run train_classifier() before evaluating the model.")

        y_pred = self.model.predict(self.X_test_pca)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(
            self.y_test,
            y_pred,
            labels=list(CLASS_TO_LABEL.values()),
            target_names=self.class_names,
            zero_division=0,
        )
        conf_matrix = confusion_matrix(
            self.y_test,
            y_pred,
            labels=list(CLASS_TO_LABEL.values()),
        )
        return accuracy, report, conf_matrix, y_pred
