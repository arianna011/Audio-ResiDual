from torch.utils.data import DataLoader, Dataset, random_split
from .processing import AudioProcessing
from .download_utils import *

class AudioDataset(Dataset):

    def __init__(self, df, data_path, sr=None, duration=None, n_chan=None, augment=False):
        """
        Params:
            df - dataframe containing audio filenames and corresponding class labels
            data_path - path to the folder containing the audio files
            sr - sample rate of audio files
            duration - length of audio files 
            n_chan - number of audio channels (1=mono, 2=stereo)
            augment - whether to perform data augmentation (time shift on raw audio, masking on spectograms)
        """
        super().__init__()
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


if __name__ == '__main__':
    dataset = AudioDataset(get_dataframe('ESC_50'), ESC_50_AUDIO_DIR)
    dl = DataLoader(dataset, batch_size=16, shuffle=False)
    for batch in dl:
        print(len(batch))
        print((batch[0]).shape)
        print((batch[1]).shape)
        break