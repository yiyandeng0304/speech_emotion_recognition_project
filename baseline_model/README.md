# this document is edited by yiyan 18/05

1. please update ONLY in the current work structure
2. comment in the beginning of each document
3. follow a certain comment format/style
4. make credits: yiyan/alina/marco/ai(claude, chatgpt) in critical parts
5. make sure to understand what we are doing, what you're doing, and that everyonw KNOWS it!
```
 Project_Root
├──  code/                       # Any scripts written by us
│   ├──  config.py               # - all** 
│   ├──  baseline_model.py       # - ?
│   ├──  data_preprocess.py      # - yiyan
│   ├──  feature_extractor.py    # - marco
│   ├──  metadata.py             # - alina
│   ├──  packages.txt            # - ?
│   ├──  run_pipeline.py         # - ?
│   └──  train.py                # - ?
├── data/
│   ├── raw/
│   │   ├── CREMADAudioWAV/            # Raw dataset with .wav files. IMPORTANT: download from kaggel**      
│   │   └── VideoDemographics.csv      # csv file with auditional metadata**
    └──  processed_data/
        ├──  features/           # Extracted features
        ├──  models/             # Saved model checkpoints
        ├──  processed_audio/    # Preprocessed audio files
        └──  results/            # Evaluation results & plots
    └── metadata_cremad.csv 
```

- leave a comment and update in here
Alina, proposal: 
Code folder: 
1. add config.py - this is a file which stores all of the variables Paths: raw audio directory, processed audio directory, metadata CSV path, features directory, models directory, results directory. 
Audio Preprocess Parameters: target sample rate, target duration (sec), silence trim threshold in dB, pre-emphasis coefficient, padding strategy (zero-pad or repeat). 
and so on. 
Why? Want to change some parameters of preprocessing? No need to search for a variable in one of the scripts. The same goes to paths which are generally a pain to configure. 
2. POTENTIALLY: split baseline_model.py into split.py, train.py, evaluate.py - simply more transparency 

Data folder: 
1. separate data into row and preprocessed. 
2. Add demographics.csv (CREMAD) file for metadata extraction 
3. Add metadata_cremad.csv after running metadata.py
4. Potentially: separate features into features_train.csv, features_eval.csv, features.test.csv - especially if we decide with code separation mentioned above 
5. also potentially: save the best model into best_model.pkl file and store it in data folder

** config.py contains all of the variables used thorughout the pipeline. If you want to have a different folder structure, you can configure 
paths to all of the audios (raw/processed), metadata, etc in this file. Also you can configure variable set-up for pre-processing in this file. 
** CREMAD audio files must be downloaded from kaggle (https://www.kaggle.com/datasets/ejlok1/cremad?resource=download), github download will break data_preprocess.py, files downloaded from githab are not actually .WAV so the script cannot process them 
** VideoDemographics.csv can be downloaad from github 



