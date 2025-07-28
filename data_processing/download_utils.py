import os
import urllib.request
import zipfile
import tarfile
import pandas as pd


# ESC-50 dataset
ESC_50_URL = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
ESC_50_OUT = 'data/esc50.zip'
ESC_50_AUDIO_DIR = 'data/esc50/ESC-50-master/audio/'
ESC_50_META_FILE = 'data/esc50/ESC-50-master/meta/esc50.csv'
ESC_50_COLUMNS = {"file_column": "filename", "label_column": "target", "fold_column":"fold"}
ESC_50_AUDIO_FOLDS = 5
ESC_50_AUDIO_LEN = 5 # 5 seconds long clips
ESC_50_CLASS_LABELS = [
    'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects',
    'sheep', 'crow', 'rain', 'sea_waves', 'crackling_fire', 'crickets',
    'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush',
    'thunderstorm', 'crying_baby', 'sneezing', 'clapping', 'breathing',
    'coughing', 'footsteps', 'laughing', 'brushing_teeth', 'snoring',
    'drinking_sipping', 'door_wood_knock', 'mouse_click', 'keyboard_typing',
    'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner',
    'clock_alarm', 'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw',
    'siren', 'car_horn', 'engine', 'train', 'church_bells', 'airplane',
    'fireworks', 'hand_saw'
]

# UrbanSound8K
URBAN_SOUND_URL = 'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz'
URBAN_SOUND_OUT = 'data/urbansound.tar.gz'
URBAN_SOUND_AUDIO_DIR = 'data/urbansound/UrbanSound8K/audio/' # + folds
URBAN_SOUND_META_FILE = 'data/urbansound/UrbanSound8K/metadata/UrbanSound8K.csv'
URBAN_SOUND_COLUMNS = {"file_column": "slice_file_name", "label_column": "classID", "fold_column":"fold"}
URBAN_SOUND_FOLDS = 10
URBAN_SOUND_AUDIO_LEN = (1,4) # variable length from 1 to 4 seconds
URBAN_SOUND_CLASS_LABELS = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

DATASETS = {  
    'ESC50': {'url': ESC_50_URL, 'audio_dir': ESC_50_AUDIO_DIR, 'csv_path': ESC_50_META_FILE, 'out_dir': ESC_50_OUT, 'class_labels':ESC_50_CLASS_LABELS, 'columns': ESC_50_COLUMNS,
              'n_folds': ESC_50_AUDIO_FOLDS, 'audio_len': ESC_50_AUDIO_LEN}, 
            
    'UrbanSound8K': {'url': URBAN_SOUND_URL, 'audio_dir': URBAN_SOUND_AUDIO_DIR, 'csv_path': URBAN_SOUND_META_FILE, 'out_dir': URBAN_SOUND_OUT,'class_labels': URBAN_SOUND_CLASS_LABELS, 'columns': URBAN_SOUND_COLUMNS, 'n_folds': URBAN_SOUND_FOLDS, 'audio_len': URBAN_SOUND_AUDIO_LEN}}


def download_dataset(url, dest_path):
    """
    Downloads a dataset from the given URL in the specified destination path.
    If the URL points to a compressed archive (zip, tar.gz), the content is extracted.
    Skips download if the file already exists.

    Params:
        url - the dataset to download
        dest_path - the destination file path
    Returns:
        extract_path - path to the folder where dataset files reside (files are extracted if the dataset is an archive)
    """

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if not os.path.exists(dest_path):
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    else:
        print(f"File already exists at {dest_path}")

    if dest_path.endswith(".tar.gz") or dest_path.endswith(".tgz"):
        extract_dir = dest_path.replace(".tar.gz", "").replace(".tgz", "")
    elif dest_path.endswith(".zip"):
        extract_dir = dest_path.replace(".zip", "")
    else:
        return dest_path

    if not os.path.exists(extract_dir):

        os.makedirs(extract_dir)
        if dest_path.endswith(".zip"):
            with zipfile.ZipFile(dest_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print("ZIP extraction complete.")

        elif dest_path.endswith(".tar.gz") or dest_path.endswith(".tgz"):
            with tarfile.open(dest_path, "r:gz") as tar_ref:
                tar_ref.extractall(extract_dir)
            print("TAR.GZ extraction complete.")
    else:
        print("Already extracted.")

    return extract_dir


def get_dataframe(dataset_name, cwd="./", downloaded=False):
    """
    Get a pandas dataframe containing:
     - filename of the audio waveform
     - label of the corresponding class
     - fold to which the file belongs
    for the specified dataset name
    considering the given path to the current execution folder
    """

    assert dataset_name in DATASETS.keys(), f"Dataset not recognized: {dataset_name}"

    dataset = DATASETS[dataset_name]
    out_path = os.path.join(cwd, dataset["out_dir"])
   
    if not downloaded and not os.path.exists(out_path):
            download_dataset(dataset["url"], out_path)
    
    df = pd.read_csv(os.path.join(cwd, dataset["csv_path"]))

    return process_dataframe(df, dataset_name)


def process_dataframe(df, dataset_name):

    cols = DATASETS[dataset_name]["columns"]
    if dataset_name == "UrbanSound8K":
       df[cols["file_column"]] = df.apply(lambda row: os.path.join(f"fold{row.fold}", row.slice_file_name), axis=1 )

    # Standardize output
    df = df.rename(columns={cols["file_column"]: "filename", cols["label_column"]: "target"})
    if "fold_column" in cols.keys():
        df = df.rename(columns={cols["fold_column"]: "fold"})
        return df[["filename", "target", "fold"]]
    return df[["filename", "target"]]

if __name__ == '__main__':
    download_dataset(URBAN_SOUND_URL, URBAN_SOUND_OUT)