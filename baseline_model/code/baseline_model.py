"""
Yiyan: baseline_model.py
Baseline: SVM model for speech emotion recognition.
strcuture of the script: classes for baseline model, including a base class and an SVM subclass.
input: extracted features (88 eGeMAPS) and labels (emotion categories) for training and evaluation.
       (not the audio file itself)
output: predicted emotion labels and evaluation metrics.
        pkl files for the trained model, scaler, and label encoder.
notes: this only works for models that work on the extracted features, not end-to-end models that take raw audio as input.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from abc import ABC, abstractmethod

# configuring logger for warning messages
logging.basicConfig (
    level = logging.INFO, 
    format = "%(levelname)s - %(message)s"
)

class BaselineModel:
    """
    Base class for emotion recognition models.
    """
    def __init__(self, model_name='baseline'):
        """
        Initialize baseline model.
        
        Args:
            model_name (str): Name of the model
        """
        self.model_name = model_name 
        self.model = None # placeholder for the actual model (e.g., SVM, CNN, etc.), see below for the inherited SVMModel class
        self.scaler = StandardScaler() # for feature standardization since the 88 eGeMAPS features have different scales
        self.label_encoder = LabelEncoder() # for encoding emotion labels to integers
        self.is_trained = False
        
    #for models that work on the extracted features:
    def preprocess_features(self, X, fit=False):
        """
        Preprocess features (standardization).
        
        Args:
            X (np.array): Feature matrix
            fit (bool): Whether to fit the scaler
            
        Returns:
            np.array: Preprocessed features
        """
        if fit: #fit means that to compute the mean and std of the training data to transform the training data.
            X_scaled = self.scaler.fit_transform(X) 
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def encode_labels(self, y, fit=False):
        """
        Encode the emotion labels (e.g. ['happy', 'angry', 'neutral'] to integers [0, 1, 2].
        
        Args:
            y (np.array): Label array
            fit (bool): Whether to fit the encoder
            
        Returns:
            np.array: Encoded labels
        """
        if fit:
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)
        
        return y_encoded
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
        """
        # an abstract methods, implemention followed in the subclass
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (np.array): Feature matrix
            
        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.preprocess_features(X, fit=False)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): True labels
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        report_dict = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        # print the evaluation results for now:
        print("\n" + "="*60)
        print(f"Baseline model evaluation: {self.model_name.upper()}")
        print(f"Accuracy : {accuracy:.2%}")
        print("="*60)
        print("Details:")
        # with precision, recall, f1 score
        print(classification_report(y, y_pred))
        print("="*60 + "\n")
        return {
            'accuracy': accuracy,
            'classification_report': report_dict,
            'confusion_matrix': cm
        }
    
    def save_model(self, save_dir):
        """
        Save model pkl. files
        
        Args:
            save_dir (str): Directory to save model
        """
        os.makedirs(save_dir, exist_ok=True)
        # save the model, scaler, and label encoder as SEPARATE pkl files
        model_path = os.path.join(save_dir, f'{self.model_name}_model.pkl')
        scaler_path = os.path.join(save_dir, f'{self.model_name}_scaler.pkl')
        encoder_path = os.path.join(save_dir, f'{self.model_name}_encoder.pkl')
        # separating the model, scaler, and label encoder allows for more flexibility in loading and using them
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Model saved to {save_dir}")
    
    def load_model(self, save_dir):
        """
        Load model.
        
        Args:
            save_dir (str): Directory containing saved model
        """
        model_path = os.path.join(save_dir, f'{self.model_name}_model.pkl')
        scaler_path = os.path.join(save_dir, f'{self.model_name}_scaler.pkl')
        encoder_path = os.path.join(save_dir, f'{self.model_name}_encoder.pkl')
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.is_trained = True # since we are loading a trained model, we can set this to True directly.
        # to prevent data leakage, we will not allow the loaded model to be further trained. 
        # if we need to train a new model, they can initialize a new class.
        
        print(f"Model loaded from {save_dir}")

# the SVMModel class inherits from BaselineModel and implements the train method using SVM.
# SVM works on the features.
class SVMModel(BaselineModel):
    """
    SVM-based emotion recognition model. 
    for current CREMAD dataset, we will use RBF kernel with default parameters (C=1.0, gamma='scale') as a baseline.
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', model_name='svm_baseline'):
        """
        Initialize SVM model.
        
        Args:
            kernel (str): Kernel type ('linear', 'rbf', 'poly')
            C (float): Regularization parameter
            gamma (str or float): Kernel coefficient
            model_name (str): Name of the model
        """
        super().__init__(model_name=model_name)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=42,
            probability=True  # Enable probability estimates
        )
    
    def train(self, X_train, y_train):
        """
        Train the SVM model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
        """
        print(f"Training SVM model (kernel={self.kernel}, C={self.C})...")
        
        # Preprocess features
        X_train_scaled = self.preprocess_features(X_train, fit=True)
        
        # Encode labels
        y_train_encoded = self.encode_labels(y_train, fit=True)
        
        # Train model
        self.model.fit(X_train_scaled, y_train_encoded)
        self.is_trained = True
        
        print("Training complete")
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (np.array): Feature matrix
            
        Returns:
            np.array: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.preprocess_features(X, fit=False)
        probas = self.model.predict_proba(X_scaled)
        
        return probas
    
    def get_model_info(self):
        """
        Get model information.
        
        Returns:
            dict: Model parameters and info
        """
        return {
            'model_type': 'SVM',
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'num_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist()
        }


if __name__ == "__main__":
    import config
    print("Testing BaselineModel and SVMModel with random data") # not on the actural dataset, just to test the code
    
    # test the model with random data
    X_dummy = np.random.randn(100, 88)
    y_dummy = np.random.choice(['happy', 'anger', 'neutral'], 100)
    
    model = SVMModel(kernel='rbf')
    model.train(X_dummy, y_dummy)
    model.evaluate(X_dummy, y_dummy)
    # test prediction and save functions
    test_dir = config.ROOT / "models" / "test"
    model.save_model(str(test_dir))
    print("unit test with random data passed")