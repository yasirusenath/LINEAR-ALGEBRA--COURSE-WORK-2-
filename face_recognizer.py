import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

class FaceRecognizerWrapper:
    """
    Wrapper class encapsulating data loading, preprocessing, PCA dimensionality reduction,
    and Logistic Regression classification for face recognition.
    """
    def __init__(self, img_size=300, n_components=50, seed=42):
        self.img_size = img_size
        self.n_components = n_components
        self.seed = seed
        self.class_names = ['student_1', 'student_2', 'student_3']
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.pca = PCA(n_components=self.n_components, random_state=self.seed)
        
        # Data containers
        self.X = None
        self.y = None
        self.image_paths = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_train_pca, self.X_test_pca = None, None

    def load_and_preprocess_data(self, DATASET_DIR):
        """Reads images, resizes, flattens into X matrix, and creates label array y."""
        dataset_path = Path(DATASET_DIR)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif'}
        
        data_list = []
        labels_list = []
        path_list = []
        
        for idx, class_name in enumerate(self.class_names):
            class_folder = dataset_path / class_name
            if not class_folder.exists():
                print(f"Warning: Folder {class_folder} does not exist!")
                continue
            
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_resized = cv2.resize(img, (self.img_size, self.img_size))
                        img_flattened = img_resized.flatten()
                        
                        data_list.append(img_flattened)
                        labels_list.append(idx)
                        path_list.append(str(img_path))
                        
        # Safeguard: Raise a clear error if no images were loaded
        if len(data_list) == 0:
            raise ValueError(
                f"Error: No images were loaded from '{dataset_path}'. "
                f"Please ensure folders 'student_1', 'student_2', and 'student_3' exist inside this directory."
            )
                        
        self.X = np.array(data_list, dtype=np.float32) / 255.0  
        self.y = np.array(labels_list, dtype=np.int32)
        self.image_paths = path_list
        print(f"Successfully loaded {self.X.shape[0]} total images with feature shape {self.X.shape[1]}")

    def split_data(self, test_size=0.20):
        """Performs a stratified train/test split."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.seed, stratify=self.y
        )
        print(f"Training set shape: {self.X_train.shape}, Test set shape: {self.X_test.shape}")

    def apply_pca_and_train(self):
        """Fits PCA on training data, transforms features, and trains Logistic Regression."""
        self.X_train_pca = self.pca.fit_transform(self.X_train)
        self.X_test_pca = self.pca.transform(self.X_test)
        
        print(f"Training PCA shape: {self.X_train_pca.shape}")
        
        # Train model
        self.model.fit(self.X_train_pca, self.y_train)
        print("Model training completed successfully!")

    def evaluate(self):
        """Predicts and returns accuracy, report, and predictions."""
        y_pred = self.model.predict(self.X_test_pca)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=self.class_names)
        cm = confusion_matrix(self.y_test, y_pred)
        return acc, report, cm, y_pred