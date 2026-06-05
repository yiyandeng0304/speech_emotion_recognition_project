from pathlib import Path 

# paths 
_CONFIG_FILE = Path(__file__).resolve()
# project root 
ROOT = _CONFIG_FILE.parent.parent

RAW_AUDIO_DIR = ROOT / "data" / "raw" / "CREMADAudioWAV" 
DEMO_PATH = ROOT / "data" / "raw" / "VideoDemographics.csv"
METADATA_PATH = ROOT / "data" / "metadata_crema.csv"
PROCESSED_AUDIO_DIR = ROOT / "data" / "processed" / "AudioCREMAD"
FEATURES_DIR = ROOT / "data" / "processed"/"features" 
RESULTS_DIR = ROOT / "data" / "processed" / "results"

# Audio processing 
SAMPLE_RATE = 16000 
DURATION = 3.0 
TOP_DB = 20 #standart for clean speech recordings
PRE_EMPHASIS_COEF = 0.95