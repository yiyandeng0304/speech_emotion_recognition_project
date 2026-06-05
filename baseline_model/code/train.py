"""
Yiyan: train.py
Training pipeline for CREMA-D emotion recognition.
Splits data by speaker ID and trains SVM baseline model.
input: features_cremad.csv (extracted features and labels for CREMA-D)
output: predicted emotion labels and evaluation metrics (accuracy, classification report, confusion matrix) for validation and test sets.
structure: OOP (CREMADTrain Class Pipeline)
comments: Robust path handling and unified sklearn evaluation matrices applied.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from baseline_model import SVMModel
import config
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# configuring logger for warning messages
logging.basicConfig (
    level = logging.INFO, 
    format = "%(levelname)s - %(message)s"
)

class CREMADTrain:
    """
    Training pipeline for CREMA-D dataset.
    Handles data splitting, training, evaluation, and logging.
    """
    
    def __init__(self, features_path, emotions=['happy', 'neutral', 'anger'], 
                 dev_ratio=0.80, val_ratio=0.10, test_ratio=0.10, random_state=42):
        """
        Initialize trainer.
        
        Args:
            features_path (str): Path to features CSV file
            emotions (list): List of emotions to use
            dev_ratio (float): Ratio of speakers for development set
            val_ratio (float): Ratio of speakers for validation set
            test_ratio (float): Ratio of speakers for test set
            random_state (int): Random seed
        """
        self.features_path = Path(features_path)
        self.emotions = emotions
        self.dev_ratio = dev_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        self.features_df = None
        self.splits = {}
        self.results = {}
    

    def load_features(self):
        """
        Load features from CSV file.
        """
        print("Loading features from :", self.features_path)
        # Load features and extract speaker IDs from file names
        self.features_df = pd.read_csv(self.features_path)
        
        if 'file_id' not in self.features_df.columns:
            raise KeyError("Error: 'file_id' is missing from the csv file.")
        # Filter for specified emotions
        self.features_df = self.features_df[
            self.features_df['emotion_label'].isin(self.emotions)
        ].copy() # to be safe for future modifications without SettingWithCopyWarning
        
        # we have to split the file_id to get the speaker_id, because the original features CSV does not have a separate speaker_id column.
        if 'speaker_id' not in self.features_df.columns:
            print("Warning: 'speaker_id' column missing in CSV. Extracting from 'file_id'")
            self.features_df['speaker_id'] = self.features_df['file_id'].apply(lambda x: str(x).split('_')[0])
        else:
            # if speaker_id column already exists, we can just use it. (in case for the fix in feature extractor)
            self.features_df['speaker_id'] = self.features_df['speaker_id'].astype(str)
        
        print(f"Loaded {len(self.features_df)} samples")
        print(f"Analyzed emotions: {self.emotions}")
        print(f"\n Emotion distribution:")
        print(self.features_df['emotion_label'].value_counts())
        
        return self.features_df
    
    def split_data_by_speaker(self):
        """
        Split data into dev/val/test sets based on speaker IDs.
        Ensures no speaker appears in multiple splits.

        Returns:
            dict: Dictionary containing split information
        """
        print("\nSplitting data by speaker ID...")
        
        # Get unique speaker IDs
        speaker_ids = self.features_df['speaker_id'].unique()
        n_speakers = len(speaker_ids)
        
        print(f"Total speakers: {n_speakers}")
        
        # Calculate number of speakers for each split
        n_val = int(n_speakers * self.val_ratio)
        n_test = int(n_speakers * self.test_ratio)
        n_dev = n_speakers - n_val - n_test  # Adjust dev to ensure total numberis correct
        
        print(f"Development speakers: {n_dev} ({self.dev_ratio*100:.1f}%)")
        print(f"Validation speakers: {n_val} ({self.val_ratio*100:.1f}%)")
        print(f"Test speakers: {n_test} ({self.test_ratio*100:.1f}%)")
        
        # Shuffle and split speaker IDs
        np.random.seed(self.random_state)
        shuffled_speakers = np.random.permutation(speaker_ids)
        
        dev_speakers = shuffled_speakers[:n_dev]
        val_speakers = shuffled_speakers[n_dev:n_dev+n_val]
        test_speakers = shuffled_speakers[n_dev+n_val:]
        
        # Split data based on speaker IDs
        dev_data = self.features_df[self.features_df['speaker_id'].isin(dev_speakers)].copy()
        val_data = self.features_df[self.features_df['speaker_id'].isin(val_speakers)].copy()
        test_data = self.features_df[self.features_df['speaker_id'].isin(test_speakers)].copy()
        
        # Get feature columns (exclude metadata), because these are not used for training.
        non_feature_cols = ['uuid', 'file_id', 'filename', 'dataset', 'speaker_id', 
                            'sentence_id', 'emotion_raw', 'emotion_label', 'intensity', 
                            'gender', 'age', 'orig_sample_rate', 'orig_duration_s', 
                            'proc_sample_rate', 'proc_duration_s', 'silence_trimmed_s', 
                            'split', 'split_strategy']
        
        feature_cols = [
            col for col in self.features_df.columns 
            if (col not in non_feature_cols) and pd.api.types.is_numeric_dtype(self.features_df[col])
        ]
        
        # Extract features and labels
        X_dev = dev_data[feature_cols].values.astype(np.float32)
        y_dev = dev_data['emotion_label'].values
        
        X_val = val_data[feature_cols].values.astype(np.float32)
        y_val = val_data['emotion_label'].values
        
        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data['emotion_label'].values
        
        # Store splits
        self.splits = {
            'dev': {'X': X_dev, 'y': y_dev, 'speakers': dev_speakers, 'data': dev_data},
            'val': {'X': X_val, 'y': y_val, 'speakers': val_speakers, 'data': val_data},
            'test': {'X': X_test, 'y': y_test, 'speakers': test_speakers, 'data': test_data}
        }
        
        print(f"\n Data split complete")
        print(f"Development set: {len(X_dev)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return self.splits
    
    def train_model(self, kernel='rbf', C=1.0):
        """
        Train SVM model on development set.
        
        Args:
            kernel (str): SVM kernel type
            C (float): Regularization parameter
            
        Returns:
            SVMModel: Trained model
        """
        print(f"\n Training SVM model (kernel={kernel}, C={C})")
        
        # Initialize model
        model = SVMModel(kernel=kernel, C=C)
        
        # Train on development set
        model.train(self.splits['dev']['X'], self.splits['dev']['y'])
        
        return model
    
    def evaluate_model(self, model, split='val'):
        """
        Evaluate model on specified split.
        
        Args:
            model (SVMModel): Trained model
            split (str): Split to evaluate on ('val' or 'test')
            
        Returns:
            dict: Evaluation results
        """
        print(f"\nEvaluating on {split} set")
        
        X = self.splits[split]['X']
        y = self.splits[split]['y']
        
        # Get predictions
        y_pred = model.predict(X)
        # calculations:
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=model.label_encoder.classes_)
        report_dict = classification_report(y, y_pred, output_dict=True)
    
        
        # Print results
        print(f"\n{split.upper()} SET RESULTS:")
        print("="*70)
        print(f"Accuracy: {accuracy:.4f}")
        print("\n Classification Report:")
        print(classification_report(y, y_pred))
        
        # Store results
        self.results[split] = {
            'accuracy': float(accuracy), 
            'predictions': y_pred.tolist(), 
            'true_labels': y.tolist(),
            'classification_report': report_dict,
            'confusion_matrix': cm.astype(int).tolist(),
            'classes': model.label_encoder.classes_.tolist() 
        }
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm.astype(int).tolist(),
            'classification_report': report_dict
        }
    
    def plot_confusion_matrix(self, split='test', save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            split (str): Split to plot ('val' or 'test')
            save_path (str): Path to save plot
        """
        if split not in self.results:
            print(f"No results available for {split} set")
            return
        
        cm = self.results[split]['confusion_matrix']
        classes = self.results[split]['classes']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes,
                   yticklabels=classes)
        plt.title(f'Confusion Matrix - {split.upper()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            # make sure the directory exists before saving
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Confusion matrix saved to {save_path}")
        plt.close()
    
    def save_results(self, save_dir, model_info=None):
        """
        Save training results and splits information.
        
        Args:
            save_dir (str): Directory to save results
            model_info (dict): Model information
        """
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        # Save split information
        split_info = {
            'emotions': self.emotions,
            'dev_speakers': self.splits['dev']['speakers'].tolist(),
            'val_speakers': self.splits['val']['speakers'].tolist(),
            'test_speakers': self.splits['test']['speakers'].tolist(),
            'dev_samples': len(self.splits['dev']['X']),
            'val_samples': len(self.splits['val']['X']),
            'test_samples': len(self.splits['test']['X'])
        }
        with open(save_dir_path / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        # Save results
        results_summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'emotions': self.emotions,
            'model_info': model_info,
            'val_accuracy': self.results.get('val', {}).get('accuracy'),
            'test_accuracy': self.results.get('test', {}).get('accuracy'),
            'val_report': self.results.get('val', {}).get('classification_report'),
            'test_report': self.results.get('test', {}).get('classification_report')
        }
        
        with open(save_dir_path / 'results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n Results saved to {save_dir_path}")
    
    def run_training_pipeline(self, kernel='rbf', C=1.0, save_dir=None):
        """
        Run complete training pipeline.
        
        Args:
            kernel (str): SVM kernel type
            C (float): Regularization parameter
            save_dir (str): Directory to save results
            
        Returns:
            tuple: (model, results)
        """
        print("\n" + "="*70)
        print("CREMA-D EMOTION RECOGNITION TRAINING PIPELINE")
        print("="*70)
        
        # Load features
        self.load_features()
        
        # Split data
        self.split_data_by_speaker()
        
        # Train model
        model = self.train_model(kernel=kernel, C=C)
        
        # Evaluate on validation set
        self.evaluate_model(model, split='val')
        
        # Evaluate on test set
        self.evaluate_model(model, split='test')
        
        # Plot confusion matrices
        if save_dir:
            save_dir_path = Path(save_dir)
            self.plot_confusion_matrix('val', save_dir_path / 'confusion_matrix_val.png')
            self.plot_confusion_matrix('test', save_dir_path / 'confusion_matrix_test.png')
        # Save results
        if save_dir:
            model_info = model.get_model_info()
            self.save_results(save_dir, model_info)
            model.save_model(str(save_dir))
        
        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print(f"Validation Accuracy: {self.results['val']['accuracy']:.4f}")
        print(f"Test Accuracy: {self.results['test']['accuracy']:.4f}")
        print("="*70 + "\n")
        
        return model, self.results


# Main execution
if __name__ == "__main__":
    # Configuration
    
    FEATURES_PATH = config.FEATURES_DIR/ "features_cremad.csv"
    RESULTS_DIR = config.RESULTS_DIR
    
    # Initialize trainer
    trainer = CREMADTrain(
        features_path=FEATURES_PATH,
        emotions=['happy', 'neutral', 'anger'],
        dev_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        random_state=42
    )
    
    # Run training pipeline
    model, results = trainer.run_training_pipeline(
        kernel='rbf',
        C=1.0,
        save_dir=RESULTS_DIR
    )
    
    print("\n Check results in:", RESULTS_DIR)