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

DATASETS = {'ESC50': {'url': ESC_50_URL, 'audio_dir': ESC_50_AUDIO_DIR, 'csv_path': ESC_50_META_FILE, 'out_dir': ESC_50_OUT}}


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


def get_dataframe(dataset_name, cwd="./"):
    """
    Get a pandas dataframe containing two columns:
     - filename of the audio waveform
     - label of the corresponding class
    for the specified dataset name
    considering the given path to the execution folder
    """

    assert dataset_name in DATASETS.keys, "Dataset not recognized: " + dataset_name

    dataset = DATASETS[dataset_name]
    out_path = os.path.join(cwd, dataset["out_dir"])
   
    if not os.path.exists(out_path):
            download_dataset(dataset["url"], out_path)
    
    df = pd.read_csv(os.path.join(cwd, dataset["csv_path"]))
    return df[['filename', 'target']]
        

if __name__ == '__main__':
    df = get_dataframe("ESC50")
    print(df.head())