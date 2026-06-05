
"""
Alina: metadata.py 
input: audio files (CREMA_D), other files (here, demographics.csv)
output: csv file with metadata
"""


from pathlib import Path 
import re 
import pandas as pd 
import logging 
import config

# These dictionaries are meant to bring into unity potential differences in lables in names
# emotion map dictionary: mapping coded emotions in filenames to proper emotion labels 
emotion_map = {"ANG":"anger", "DIS":"disgust", "FEA": "fear", "HAP": "happy", "NEU":"neutral", "SAD":"sad"}

# emotion level map dictionary: mapping coded emotion level in filenames to full emotion level 
emotion_level_map = {"LO":"low", "MD":"medium", "HI":"high", "XX":"unspecified"}

# regex for parsing the CREMAD filename 
# actor_id: 4 digits || sentence_id: 3 letters || emotion_code: 3 letters || emotion_level: 2 letters
filename_cremad_re = re.compile(r"(\d{4})_([A-Z]+)_([A-Z]+)_([A-Z]+)\.wav")

# configuring logger for warning & info messages
logging.basicConfig (
    level = logging.INFO, 
    format = "%(levelname)s - %(message)s"
)

class BaseMetadataParser():
    """
    Initialize a BASE class for metadata parser 
    this can be used as a basic parser for any data (ideally for other datasets in the future)
    any dataset-specific logic (e.g., parsing non-audio files) can be implemented in subclasses.
    """
    # dictionary schema to be used for metadata compilation 
    BASE_SCHEMA = {
    "uuid":              None,
    "file_id":           None,
    "filename":          None,
    "dataset":           None,
    "speaker_id":        None,
    "sentence_id":       None,
    "emotion_raw":       None,
    "emotion_label":     None,
    "intensity":         None,
    "gender":            None,
    "age":               None,
    "orig_sample_rate":  None,
    "orig_duration_s":   None,
    "proc_sample_rate":  None,
    "proc_duration_s":   None,
    "silence_trimmed_s": None,
    "split":             None,
    "split_strategy":    None,
    }
    # accepts paths: raw audio path, demographics file path,  
    def __init__(self, raw_audio_path, demo_path, output_path):
        """
        Args:
            raw_audio_path: path to raw audio files, global variable configured in config.py
            demo_path: path to other, non-audio files (here demographics. csv), global variable configured in config.py
            ouput_path: path to save metadata file, global variable configured in config.py
        """
        self.audio_path = Path(raw_audio_path) # raw audio path, wrap in Path so that other methods work with Path object & not str 
        self.demo_path = Path(demo_path) # demographic file path
        self.output_path =  Path(output_path) # output file path 


    def _validate_directory(self, path):
        """
        Take path and validate if it exists

        Args: 
            path: path to any file needed validation
        """
        # check if directory exists and if there are audio files in it  
        if not Path (path).is_dir():
            logging.warning (f"The derictory {path} does not exist. Please refer at the README.md for derictory clarification")
            raise FileNotFoundError 
        if not any (Path(path).glob("*.wav")):
            logging.warning (f": There are no audio files in the directory: {path}. Please refer to the README.md for download clarification")
            raise FileNotFoundError
    
    def _parse_single_file (self, path):
        """
        This method must be configured by dataset subclass due to differences in naming/other structure  
        """
        raise NotImplementedError ("_parse_single_file must be implemented by the subclass")


    def _load_demographics (self):
        """
        Same as parse)single_file but for additional non-audio files 
        """
        raise NotImplementedError ("_load_demographics must be implemented by the subclass")

    def build(self):
        """
        Compiling all together data from audio names + some addition csv (here Demographics.csv)
        """
        self._validate_directory (self.audio_path)
        skipped = [] # skipped audio files due to mismatch in name (or other) structure 
        parsed = [] # successfully parsed 
        for audio in sorted (self.audio_path.glob("*.wav")): # sorted to guarantee consistent iteration across all OS
            result = self._parse_single_file(audio)
            if result is None: 
                skipped.append (audio.name)
            else: 
                parsed.append (result)
        # info about of skipped and parsed audios
        if len(skipped) != 0: 
            logging.warning (f"Skipped {len(skipped)} files: {skipped}")
        if len(parsed) == 0:
            logging.warning (f"0 files parsed! Something went wrong")
            raise ValueError 
        logging.info (f"Records successfully parsed: {len(parsed)}")
        # interim dataframe  
        meta_df = pd.DataFrame(parsed)
        # get additional files(here, demograpgics.csv)
        demo_df = self._load_demographics()
        # merge on speaker id 
        meta_df = meta_df.merge(
            demo_df,
            on = 'speaker_id',
            how = 'left'
        )
        logging.info (meta_df['emotion_label'].value_counts().to_string())
        return meta_df


    def save (self, meta_df):
        """
        Save the final dataframe into a csv file 

        Args: 
            meta_df: dataframe resulting from processing audio files & demographics file
        
        Returns:
            csv file with metadata 
        """
        self.output_path.parent.mkdir (parents = True, exist_ok = True)
        meta_df.to_csv (self.output_path, index = False)
        logging.info (f"Metadata saved -> {self.output_path}")
        


class CREMADMetadataParser (BaseMetadataParser):
    """
    Specialized parser for CREMA-D dataset.
    Handles CREMA-D specific file structure and naming conventions.
    """
    def _parse_single_file (self, path):
        """
        Parse a single file extracting data from the name

        Args:
            path: path to raw audio files
        
        Returns: 
            a row with all gathered data (later to be part of dataframe)
        """
        match = re.search(filename_cremad_re, str(path.name))
        if not match:
            return None 
        actor_id, sentence_id, emotion_code, emotion_level = match.groups()
        emotion_label =  emotion_map[emotion_code]
        emotion_intensity = emotion_level_map[emotion_level]
        row = self.BASE_SCHEMA.copy() 
        row['uuid'] = "crema_d_" + path.stem
        row['file_id'] = path.stem
        row['filename'] = path.name 
        row['dataset'] = "crema_d" 
        row['speaker_id'] = actor_id
        row['sentence_id'] = sentence_id
        row['emotion_raw'] = emotion_code
        row['emotion_label'] = emotion_label
        row['intensity'] = emotion_intensity
        return row 


    def _load_demographics(self):
        """
        Parse additional files with data, here "demographics.csv"
        
        Returns: 
            dataframe with additional information 
        """
        if not self.demo_path.is_file():
            logging.warning ("Demographics file does not exist")
            return pd.DataFrame (columns=["speaker_id", "gender", "age"])
        df = pd.read_csv(self.demo_path)
        df = df.rename(columns = {'ActorID':'speaker_id', 'Sex':'gender', 'Age':'age'})
        df['speaker_id'] = df['speaker_id'].astype(str)
        df['gender'] = df['gender'].str.strip().str.lower()
        return df[["speaker_id", "gender", "age"]]
        

if __name__ == '__main__':
    
    parser = CREMADMetadataParser (
        raw_audio_path = config.RAW_AUDIO_DIR,
        demo_path = config.DEMO_PATH,
        output_path = config.METADATA_PATH
    )
    meta_df = parser.build()
    parser.save(meta_df)
