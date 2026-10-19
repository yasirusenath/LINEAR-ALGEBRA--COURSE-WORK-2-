
import cv2
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class FaceRecognizerWrapper:
    """
    Wrapper class for face recognition using:
    1. Image loading and preprocessing
    2. Train/test splitting
    3. PCA (Eigenfaces)
    4. Logistic Regression
    5. Model evaluation
    """

    def __init__(self, img_size=100, n_components=50, seed=42):

        self.img_size = img_size
        self.n_components = n_components
        self.seed = seed

        # Use the actual folder names as class labels.
        self.class_names = []

        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=2000
        )

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

        dataset_path = Path(DATASET_DIR)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {dataset_path}"
            )

        # Automatically detect the class folders.
        self.class_names = sorted([
            folder.name
            for folder in dataset_path.iterdir()
            if folder.is_dir()
        ])

        if len(self.class_names) < 2:
            raise ValueError(
                "At least two person folders are required."
            )

        valid_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp',
            '.tif', '.tiff', '.jfif'
        }

        data_list = []
        labels_list = []
        path_list = []

        for idx, class_name in enumerate(self.class_names):

            class_folder = dataset_path / class_name
            class_image_count = 0

            for img_path in sorted(class_folder.iterdir()):

                if not img_path.is_file():
                    continue

                if img_path.suffix.lower() not in valid_extensions:
                    continue

                # Read the image in grayscale.
                img = cv2.imread(
                    str(img_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if img is None:
                    print(f"Warning: Cannot read {img_path}")
                    continue

                # Resize all images to the same dimensions.
                img_resized = cv2.resize(
                    img,
                    (self.img_size, self.img_size)
                )

                # Flatten the image into a feature vector.
                img_flattened = img_resized.flatten()

                data_list.append(img_flattened)
                labels_list.append(idx)
                path_list.append(str(img_path))

                class_image_count += 1

            print(
                f"Class '{class_name}': "
                f"{class_image_count} images loaded"
            )

        # Check BEFORE accessing the shape of an empty array.
        if len(data_list) == 0:
            raise ValueError(
                f"No images were loaded from {dataset_path}. "
                "Check the folder structure and image files."
            )

        self.X = np.asarray(data_list, dtype=np.float32) / 255.0
        self.y = np.asarray(labels_list, dtype=np.int32)
        self.image_paths = path_list

        # Check that every class has enough images.
        for idx, name in enumerate(self.class_names):
            count = int(np.sum(self.y == idx))

            if count < 2:
                raise ValueError(
                    f"Class '{name}' has only {count} usable image(s). "
                    "At least two are required for a stratified split."
                )

        print("\nDataset loaded successfully.")
        print("Classes:", self.class_names)
        print("Total images:", len(self.X))
        print("Feature shape:", self.X.shape)

    def split_data(self, test_size=0.20):

        if self.X is None:
            raise ValueError(
                "Load the dataset before splitting the data."
            )

        # Each class needs at least one training and test image.
        class_counts = np.bincount(self.y)
        n_classes = len(self.class_names)
        n_samples = len(self.y)

        n_test = int(np.ceil(n_samples * test_size))
        n_train = n_samples - n_test

        if (
            n_test < n_classes
            or n_train < n_classes
            or np.min(class_counts) < 2
        ):
            raise ValueError(
                "Not enough images for a stratified train/test split. "
                "Add more images to each person folder or adjust test_size."
            )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=self.seed,
            stratify=self.y
        )

        print("Training set:", self.X_train.shape)
        print("Testing set:", self.X_test.shape)

    def apply_pca_and_train(self):

        if self.X_train is None:
            raise ValueError(
                "Run split_data() before training."
            )

        # PCA components cannot exceed the number of training
        # samples or the number of input features.
        actual_components = min(
            self.n_components,
            self.X_train.shape[0],
            self.X_train.shape[1]
        )

        self.pca = PCA(
            n_components=actual_components,
            svd_solver="full"
        )

        # Fit PCA ONLY on training data to prevent data leakage.
        self.X_train_pca = self.pca.fit_transform(
            self.X_train
        )

        self.X_test_pca = self.pca.transform(
            self.X_test
        )

        print("PCA components:", actual_components)
        print("Training PCA shape:", self.X_train_pca.shape)
        print("Testing PCA shape:", self.X_test_pca.shape)

        self.model.fit(
            self.X_train_pca,
            self.y_train
        )

        print("Model training completed successfully!")

    def evaluate(self):

        if self.X_test_pca is None:
            raise ValueError(
                "Train the model before evaluating it."
            )

        y_pred = self.model.predict(self.X_test_pca)

        acc = accuracy_score(self.y_test, y_pred)

        report = classification_report(
            self.y_test,
            y_pred,
            labels=np.arange(len(self.class_names)),
            target_names=self.class_names,
            zero_division=0
        )

        cm = confusion_matrix(
            self.y_test,
            y_pred,
            labels=np.arange(len(self.class_names))
        )

        return acc, report, cm, y_pred