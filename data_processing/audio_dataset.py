from torch.utils.data import DataLoader, Dataset, random_split
from .processing import AudioProcessing
from .download_utils import *
from torch.utils.data import DataLoader

class AudioDataset(Dataset):

    def __init__(self, name, df, data_path, sr=None, duration=None, n_chan=None, augment=False):
        """
        Params:
            name - name of the original dataset
            df - dataframe containing audio filenames, folds and corresponding class labels
            data_path - path to the folder containing the audio files
            sr - sample rate of audio files
            duration - length of audio files 
            n_chan - number of audio channels (1=mono, 2=stereo)
            augment - whether to perform data augmentation (time shift on raw audio, masking on spectograms)
        """
        super().__init__()
        self.name = name
        self.df = df
        self.data_path = data_path
        self.sr = sr
        self.duration = duration
        self.n_chan = n_chan
        self.augment = augment

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        """
        Return audio waveform and corresponding class label
        """
        audio, _ = AudioProcessing.load(self.data_path + self.df.loc[index, 'filename'])
        label = self.df.loc[index, 'target']
        return audio, label
    
    def process_item(self, audio):
        if self.sr: 
            audio = AudioProcessing.to_sample_rate(audio, self.sr)
        if self.n_chan: 
            audio = AudioProcessing.to_channels(audio, self.n_chan)
        if self.duration: 
            audio = AudioProcessing.to_length(audio, self.duration)
        if self.augment:
            audio = AudioProcessing.time_shift(audio)

        spec = AudioProcessing.to_log_mel_spectogram(audio)
        if self.augment:
            spec = AudioProcessing.time_freq_mask(spec)
        return spec
    


def get_fold_dataloaders(audio_dataset, shuffle=False, batch_size=32, num_workers=0):
    """
    Return list of (train_loader, val_loader) using the predefined folds on the given dataset.
    """
    name = audio_dataset.name
    df = audio_dataset.df
    n_folds = DATASETS[name]['n_folds']
    assert n_folds > 0, f'Dataset {name} is not arranged for k-fold validation'
    fold_col = DATASETS[name]['columns']['fold_column']

    folds = []

    for i in range(n_folds):   
        val_fold = i+1
        train_df = df[df[fold_col] != val_fold].reset_index(drop=True)
        val_df = df[df[fold_col] == val_fold].reset_index(drop=True)

        train_dataset = AudioDataset(name, train_df, audio_dataset.data_path)
        val_dataset = AudioDataset(name, val_df, audio_dataset.data_path)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        folds.append((train_loader, val_loader))

    return folds    


if __name__ == '__main__':
    dataset = AudioDataset('UrbanSound8K', get_dataframe('UrbanSound8K'), URBAN_SOUND_AUDIO_DIR)
    dl = DataLoader(dataset, batch_size=16, shuffle=False)
    folds = get_fold_dataloaders(dataset)
    print(len(folds))