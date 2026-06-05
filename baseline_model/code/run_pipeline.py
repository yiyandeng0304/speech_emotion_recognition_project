"""
Yiyan+AI: run_pipeline.py
Complete pipeline execution script for CREMA-D emotion recognition.
Coordinates audio preprocessing, feature extraction, and model training.
input: Raw audio dataset from config.RAW_AUDIO_DIR
output: Processed audio, extracted features CSV, and training evaluation results.
structure: Procedural pipeline main execution
comments: aligned with config.py paths and corrected the pipeline stage label mappings.
"""

import os
import sys
from pathlib import Path

# Add current directory to path to ensure safe relative imports
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import individual pipeline steps and config
from data_preprocess import CREMADPreprocessor
from feature_extractor import CREMADExtractor
from train import CREMADTrain
import config


def main():
    """Run complete pipeline"""
    
    print("\n" + "="*70)
    print("CREMA-D EMOTION RECOGNITION - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # ─── STEP 1: PREPROCESS AUDIO DATA ───────────────────────────────────
    print("STEP 1: Preprocessing audio data")
    print("-"*70)
    
    # this is the capitalized form of the emotion labels in the raw audio filenames, which the preprocessor will use to filter and process only the relevant files.
    raw_emotions = ['HAP', 'NEU', 'ANG']
    
    preprocessor = CREMADPreprocessor(
        target_sample_rate  =config.SAMPLE_RATE,            
        normalize=True,
        trim_silence=True,
        emotions_to_process=raw_emotions,
        target_duration=config.DURATION,
        trim_db=config.TOP_DB,
        pre_emphasis_coef=config.PRE_EMPHASIS_COEF,
        raw_audio_path=str(config.RAW_AUDIO_DIR),
        metadata_path=str(config.METADATA_PATH),
        processed_audio_path=str(config.PROCESSED_AUDIO_DIR)
    )

    print(f"Reading raw audio from: {config.RAW_AUDIO_DIR}")
    print(f"Saving processed audio to: {config.PROCESSED_AUDIO_DIR}")
    
    preprocessor.process_dataset()
    
    # ─── STEP 2: EXTRACT FEATURES ────────────────────────────────────────
    print("\nSTEP 2: Extracting features")
    print("-"*70)
    features_csv_path = config.FEATURES_DIR / "features_cremad.csv"
    target_emotions = ['happy', 'neutral', 'anger']
    extractor = CREMADExtractor(
        processed_audio_path=str(config.PROCESSED_AUDIO_DIR),
        metadata_path=str(config.METADATA_PATH),
        output_path=str(features_csv_path),
        target_emotions=target_emotions
    )
    print(f"Extracting features from: {config.PROCESSED_AUDIO_DIR}")
    print(f"Saving features CSV to: {features_csv_path}")

    extractor.extract_all()
    
    # ─── STEP 3: TRAIN MODEL ─────────────────────────────────────────────
    print("\nSTEP 3: Training model")
    print("-"*70)

    trainer = CREMADTrain(
        features_path=features_csv_path, 
        emotions=target_emotions        
    )
    model, results = trainer.run_training_pipeline(
        kernel='rbf',
        C=1.0,
        save_dir=config.RESULTS_DIR
    )
    
    # ─── PIPELINE COMPLETE ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {results['test']['accuracy']:.4f}")
    print(f"All outputs successfully generated and saved.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()