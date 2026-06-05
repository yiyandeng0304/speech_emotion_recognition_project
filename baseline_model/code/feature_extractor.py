"""
Alina: feature_extractor.py
goal: Feature extraction using OpenSMILE eGeMAPS (88 dimensions).
input: preprocessed data, metadata file
output: csv file with features

"""

import opensmile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from abc import ABC, abstractmethod
import logging
import librosa
import numpy as np


# configuring logger for warning messages
logging.basicConfig (
    level = logging.INFO, 
    format = "%(levelname)s - %(message)s"
)

class BaseFeatureExtractor (ABC):
    """
    initialize a BASE class for feature extraction. 
    this can be used as a basic extractor for any data (ideally for other datasets in the future)
    any dataset-specific logic (e.g., eGUMPS SMILE extractor) can be implemented in subclasses.
    """
    
    def __init__(self, processed_audio_path, metadata_path, output_path, target_emotions):
        """
        Initialize feature extractor.
        
        Args:
            processed_audio_path
            metadata_path
            output_path: path for a feature output file
            target_emotions: a list of emotions to proccess
        """
        self.processed_audio_path = Path(processed_audio_path)
        self.metadata_path = Path (metadata_path)
        self.output_path = Path (output_path)
        self.emotions = target_emotions

    def _validate_directory(self, path):
        """
        Take path and checks if it exists 
        
        Args: 
            file_path (str): Path to audio file, metadata or proccessed audios output path
        
        Returns:
            nothing or error  
        """
        # check if directory exists and if there are audio files in it  
        if not Path (path).is_dir():
            logging.warning (f"The derictory {path} does not exist. Please refer at the README.md for derictory clarification")
            raise FileNotFoundError 
        if not any (Path(path).glob("*.wav")):
            logging.warning (f"There are no audio files in the directory: {path}. Please refer to the README.md for download clarification")
            raise FileNotFoundError
    
    def _load_metadata (self):
        """
        Load metadata, check that file name and label columns exist,
        check that target emotions exist in metadata file  
        
        Returns: 
            dataframe with target emotions 
        """
        # check if file exists, if not, raise an error 
        if not self.metadata_path.is_file():
            logging.warning ("Metadata file does not exist")
            raise FileNotFoundError
        df = pd.read_csv(self.metadata_path)
        
        # columns that are required for building a dataframe
        # check if they exist, if not raise an error   
        required_columns = ['file_id', 'filename', 'emotion_label', 'speaker_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.warning (f"Columns {missing_columns} do not exist") 
            raise ValueError
        
        # filter by target emotions 
        df_filtered = df[df['emotion_label'].isin(self.emotions)]
        # if no target emotions exist raise an error 
        if df_filtered.empty:
            logging.warning ("Targtet emotions not found in the metadata!") 
            raise ValueError
        logging.info (f"Loaded {len(df)} total records from metadata")
        logging.info (f"Filtered {len(df_filtered)} records for emotions: {self.emotions}")
        logging.info (f"Emotion distribution:\n{df_filtered['emotion_label'].value_counts().to_string()}")
        return df_filtered

    def _load_audio(self, file_path):
        """
        Load audio file.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            tuple: (audio signal, sample rate) 
        """
        if not file_path.is_file():
            logging.warning (f"Audiofile {file_path} is not found")
            raise FileNotFoundError

        y, sr = librosa.load(file_path, sr=None, mono=True)
        return y, sr
    
    @abstractmethod
    def _extract_features(self, audio_signal, sample_rate):
        """
        Extract features from audio signal, must be implemented by subclass.
        
        Args:
            audio_signal: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Feature vector (specific to implementation)
        """
        raise NotImplementedError ("Subclass must implement _extract_features()")

    def _save_features(self, features_df):
        """
        Takes extracted features dataframe and saves it in a file. 

        Returns:
            csv file with features
        """
        self.output_path.parent.mkdir (parents = True, exist_ok = True)
        features_df.to_csv (self.output_path, index = False)
        logging.info (f"Extracted features saved -> {self.output_path}")

    def extract_all(self):
        """
        Compile all methods together 
        """
        # get the metadata dataframe 
        df_metadata = self._load_metadata()
        # check if path to processed audio exist 
        self._validate_directory(self.processed_audio_path)
        # track statistics 
        processed = 0 
        failed = 0
        features_l = []
        # loop through each file in metadata 
        for index, row in tqdm(df_metadata.iterrows()):
            file_id = row['file_id']
            filename = row ['filename']
            emotion_label = row ['emotion_label']
            speaker_id = row ['speaker_id']
            # construct path to audiofile 
            file_path = self.processed_audio_path / filename
            
            # try to proccess audio file, extract features
            # save features dic to the list
            try:
                audio_signal, sr = self._load_audio(file_path)
                features = self._extract_features (audio_signal, sr)
                pure_features_dict = features.iloc[0].to_dict()
                feature_dic = {
                    'file_id' : file_id,
                    'speaker_id' : speaker_id,
                    'emotion_label' : emotion_label,
                    **pure_features_dict  # Unpacks all features into the dictionary
                }
                features_l.append(feature_dic)
                processed += 1
            except Exception as e:
                logging.error(f"Failed to process {file_id}: {e}")
                failed += 1 
                continue

        # check if features were extracted 
        if len(features_l) == 0:
            logging.warning("No features extracted!")
            raise ValueError("Feature extraction failed for all files")
        # save features to dataframe
        features_df = pd.DataFrame(features_l)
        # log statistics 
        logging.info(f"Successfully processed: {processed}")
        logging.info(f"Failed: {failed}")
        logging.info(f"Feature dimensions: {features_df.shape}")
        # save fatures 
        self._save_features(features_df)
    



class CREMADExtractor (BaseFeatureExtractor):
    """
    Specialized extractor for CREMAD, uses openSMILE eGeMAPS feature extractor 

    """
    def __init__(self, *args, **kwargs):
        """
        Initialise CREMA-D feature extractor with openSMILE eGeMAPS
        """
        super().__init__(*args, **kwargs)
         # Initialize openSMILE extractor
        self.smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    def _extract_features(self, audio_signal, sample_rate):
        """
        Extract features with openSMILE eGeMAPS

        Args:
            audio_signal (np arraay): Audio waveform from _load_audio()
            sample_rate: Sample rate of audio _load_audio()

        Return:
            pandas Series: Feature vector with 88 eGeMAPS features  
        """
        # make sure the audio signal is in the correct format for openSMILE (float32). by yiyan on 31/05.
        audio_signal_cleaned = audio_signal.astype(np.float32)
        features = self.smile.process_signal (audio_signal_cleaned, sample_rate)
        return features


if __name__ == "__main__":
    # import config
    import config 
    # implement CREMAD extractor 
    extractor = CREMADExtractor (
        processed_audio_path=config.PROCESSED_AUDIO_DIR,
        metadata_path=config.METADATA_PATH,
        output_path=config.FEATURES_DIR / "features_cremad.csv",
        target_emotions=['happy', 'anger', 'neutral']
    )
    extractor.extract_all()