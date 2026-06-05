"""
Yiyan: data_preprocess.py
input: raw audio files from CREMA-D dataset
output: processed audio files (normalized, resampled, and standardized format) saved to a specified path.
input path: "../data/CREMAD"
output path: "../data/processed_data/processed_audio"
goal: handles audio normalization, resampling, and format standardization.
structure: a base DataPreprocessor class that provides general audio processing methods, and a CREMADPreprocessor subclass that implements dataset-specific processing logic (e.g., parsing filenames for emotion labels).
comment: make sure to stay in the right directory to run. since it caused some errors before, I added a check for the input directory. Also, the script will save a mapping of original and processed files in a CSV for reference.

Alina: slight changes in the __init__() method to capture additional preprocessing (pre_emphasis coef), addition of path structures to unify across scripts and 'universal' variables to store in config.py  
addition of logger and substitution of print statement to logger for more transparency 
addition of _validate_directory() method to validate the existance of the paths 
addition of _resampling() method,  probbaly not needed in CREMAD but useful for scaling
addition of _apply_pre_emphasis() method to balance low and hight frequencies in the signal 
addition of metadata dictionary in _preprocess_audio() method 
updated _save_audio method to capture new path structure 
updated CREMAD subclass __init__ substituted file_mapping to metadata.csv 
removed _parse_filename method, we don't need to validate and split audio file names, we can just check if it is in metadata in the process_data()
updated _process_dataset() to capture metadata update instead of dataframe/file mapping creation (mostly by AI)
"""
# packages that better are initially installed, see the packages.txt file for details.
import os
import librosa
import soundfile as sf
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from abc import ABC, abstractmethod
import config
import traceback



# configuring logger for warning messages
logging.basicConfig (
    level = logging.INFO, 
    format = "%(levelname)s - %(message)s"
)

class DataPreprocessor(ABC):
    """
    initialize a BASE class for audio data preprocessing.
    this can be used as a basic processor for any data (ideally for other datasets in the future)
    any dataset-specific logic (e.g., parsing filenames for emotion labels) can be implemented in subclasses.
    """
    
    def __init__(self, 
                 target_sample_rate, 
                 target_duration,
                 trim_db, 
                 pre_emphasis_coef,
                 raw_audio_path, 
                 metadata_path, 
                 processed_audio_path, 
                 normalize=True,
                 trim_silence=True):
        """
        Args:
            target_sample_rate (int): Target sample rate, now set to 16kHz (config.py)
            target_duration (float): Target duration in seconds (None = no padding/trimming)
            trim_db (int): Threshold for silence trimming in dB
            pre_emphasis_coef (float): must be < 1, balance low and high frequencies
            raw_audio_path (path): path for the raw audios, set in config.py
            metadata_path (path): path for the existing metadata file to update the data about audio
            processed_audio_path (path): path to put processed files for next steps 
            normalize (bool): Whether to normalize audio
            trim_silence (bool): Whether to trim silence
        """
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.normalize = normalize
        self.trim_db = trim_db
        self.pre_emphasis_coef = pre_emphasis_coef
        # since there are many files, add this attribute to keep track.
        self.stats = {
            'processed': 0,
            'failed': 0,
            'total_duration': 0
        }
        # switchers 
        self.trim_silence = trim_silence
        self.normalize = normalize
        # paths
        self.audio_path = Path(raw_audio_path) # raw audio path, wrap in Path so that other methods work with Path object & not str 
        self.processed_audio_path = Path(processed_audio_path) # output file path 
        self.metadata_path  =  Path(metadata_path)
    
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

    def _load_audio(self, file_path):
        """
        Load audio file.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            tuple: (audio_data, orig_sample_rate, orig_duration) 
            
        Comments:
        make sure of the working directory
        make sure the format of the audio files 
        """
        y, sr = librosa.load(file_path, sr=None, mono=True)
        duration = len(y) / sr
        return y, sr, duration
    
    def _resampling (self, y, orig_sr):
        """
        Do resampling to target sample rate

        Args: 
            y-array, orig_sample_rate
        
        Returns:
            resampled y-array 
        """
        # check if it's not already in the targeted sample rate
        if orig_sr == self.target_sample_rate: 
            return y
        y_resampled = librosa.resample (y, orig_sr = orig_sr, target_sr = self.target_sample_rate)
        return y_resampled
    
    # arguably, is silence an indicator of emotion?
    # if we need to keep it with bad quality audios, we can set the silence range to be very small.
    def _trim_silence(self, y):
        """
        Trim leading and trailing silence.
        
        Args:
            y (np.array): Audio signal
            
        Returns:
            np.array: Trimmed audio
            silence_trimmed_sec
        """
        if not self.trim_silence:
            return y, 0.0
        
        y_trimmed, _ = librosa.effects.trim(y, top_db=self.trim_db)
        silence_trimmed_sec = (len(y) - len(y_trimmed)) / self.target_sample_rate
        if len(y_trimmed) == 0:
            logging.warning("Silence trimming produced empty signal, returning original")
            return y, 0.0
        return y_trimmed, silence_trimmed_sec
    

    def _apply_pre_emphasis(self,y):
        """
        Adjusts low and hight frequency in the signal
        
        Args:
            y (np.array): Audio signal
            
        Returns:
            np.array: Adjusted audio
        """
        # to make sure the number of samples are correct:
        if len(y) <= 1:
            return y  # if not enough samples to apply pre-emphasis
        # from y[1:] onward has the filter applied
        return np.append (y[0], y[1:] - self.pre_emphasis_coef*y[:-1])
    
        
    def _normalize_audio(self, y):
        """
        Normalize audio to peak amplitude of 1.0.
        
        Args:
            y (np.array): Audio signal
            
        Returns:
            np.array: Normalized audio
        """
        if not self.normalize:
            return y
        
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        
        return y

    def _adjust_duration(self, y):
        """
        Adjust audio to target duration (pad or trim), if needed
        
        Args:
            y (np.array): Audio signal
            sr (int): Sample rate
            
        Returns:
            np.array: Adjusted audio
        """
        if self.target_duration is None:
            return y
        
        target_length = int(self.target_sample_rate * self.target_duration)
        current_length = len(y)
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            y = np.pad(y, (0, padding), mode='constant')
        elif current_length > target_length:
            # Trim
            start = (current_length - target_length) // 2
            end   = start + target_length
            y = y[start:end]
        return y
    
    def _preprocess_audio(self, y, sr, duration):
        """
        Apply all preprocessing steps to audio.
        
        Args:
            y (np.array): Audio signal
            sr (int): Sample rate
            
        Returns:
            np.array: Preprocessed audio
            dictionary: Gathered metadata 
        """
        # Resample 
        y = self._resampling(y, sr)

        # Trim silence
        y, silence_trimmed_s = self._trim_silence(y)

        # Apply pre-emphasis 
        y = self._apply_pre_emphasis(y)

        # Normalize
        y = self._normalize_audio(y)
        
        # Adjust duration
        y = self._adjust_duration(y)

        # dic to fill metadata with 
        data = {
            'orig_sample_rate': sr, 
            'orig_duration_s': duration, 
            'silence_trimmed_s' : silence_trimmed_s
        }
        
        return y, data 
    
    def _save_audio(self, y, sr, output_path):
        """
        Save processed audio.
        
        Args:
            y (np.array): Audio signal
            sr (int): Sample rate
            output_path (str): Output file path
            
        Returns:
            bool: Success status
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, y, sr)
            logging.info(f"Saved processed audio to: {output_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving {output_path}: {e}")
            return False
    
    def print_stats(self):
        """Print processing statistics.
        at the same time, log the stats. *changed from print to logging on 31/05."""
        logging.info("="*50)
        logging.info("       AUDIO PREPROCESSING STATISTICS")
        logging.info("="*50)
        logging.info(f" Successfully Processed : {self.stats['processed']} files")
        logging.info(f" Failed / Skipped       : {self.stats['failed']} files")
        logging.info(f" Total Audio Extracted  : {self.stats['total_duration']/3600:.2f} hours")
        logging.info("="*50)


class CREMADPreprocessor(DataPreprocessor):
    """
    Specialized preprocessor for CREMA-D dataset.
    Handles CREMA-D specific file structure and naming conventions.
    """
    
    def __init__(self, *args, emotions_to_process=None, **kwargs):
        """
        Initialize CREMA-D preprocessor.
        
        Args:
            emotions_to_process (list): List of emotions to process (e.g., ['HAP', 'NEU', 'ANG'])
                                       If None, process all emotions
        """
        super().__init__(*args, **kwargs)
        self.emotions_to_process = emotions_to_process
        self.metadata_df = pd.read_csv (self.metadata_path, index_col='file_id')

            
    def process_dataset(self):
        """
        Process CREMA-D dataset audiofiles from raw to processed directory
        Paths are set via config.py and passed through DataPreprocessor.__init__
            
        """
        # validate the input and output file paths
        self._validate_directory(self.audio_path)
        self.processed_audio_path.mkdir(parents=True, exist_ok=True)
        
        # Get all wav files
        audio_files = list(self.audio_path.glob("*.wav"))

        if self.emotions_to_process:
            audio_files = [f for f in audio_files if f.stem.split('_')[2] in self.emotions_to_process]
            logging.info(f"Filtering to emotions: {self.emotions_to_process}")

        logging.info(f"Found {len(audio_files)} files to process")
        
        # Process each file
        for filepath in tqdm(audio_files):
            # Parse filename info
            file_id = filepath.stem
            if file_id not in self.metadata_df.index:
                logging.warning(f"File '{file_id}' not found in metadata, skipping.")
                self.stats['failed'] += 1
                continue
            try:
                y, sr, duration = self._load_audio(filepath)
                y, data = self._preprocess_audio(y, sr, duration)

                output_path = self.processed_audio_path / filepath.name
                self._save_audio(y, self.target_sample_rate, output_path)

                # update metadata in-place
                self.metadata_df.loc[file_id, 'orig_sample_rate']  = data['orig_sample_rate']
                self.metadata_df.loc[file_id, 'orig_duration_s']   = data['orig_duration_s']
                self.metadata_df.loc[file_id, 'silence_trimmed_s'] = data['silence_trimmed_s']
                self.metadata_df.loc[file_id, 'proc_sample_rate']  = self.target_sample_rate
                self.metadata_df.loc[file_id, 'proc_duration_s']   = len(y) / self.target_sample_rate

                self.stats['processed'] += 1
                self.stats['total_duration'] += data['orig_duration_s']
            except Exception as e:
                logging.error(f"Failed to process '{file_id}': {e}")
                logging.error(traceback.format_exc())
                self.stats['failed'] += 1


        self.metadata_df.to_csv(self.metadata_path)
        logging.info(f"Metadata saved to {self.metadata_path}")
        logging.info(f"Processed: {self.stats['processed']} | "
                    f"Failed: {self.stats['failed']} | "
                    f"Total duration: {self.stats['total_duration']:.1f}s")


if __name__ == "__main__":
    # Initialize preprocessor for CREMA-D
    # Only process happy, neutral, and anger emotions
    preprocessor = CREMADPreprocessor(
        target_sample_rate=config.SAMPLE_RATE,
        target_duration=config.DURATION,
        trim_db=config.TOP_DB,
        pre_emphasis_coef=config.PRE_EMPHASIS_COEF,
        raw_audio_path=config.RAW_AUDIO_DIR,
        metadata_path=config.METADATA_PATH,
        processed_audio_path=config.PROCESSED_AUDIO_DIR,
        emotions_to_process=['HAP', 'NEU', 'ANG']
    )

    preprocessor.process_dataset()
    preprocessor.print_stats()