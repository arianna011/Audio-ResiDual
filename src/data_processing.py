import IPython.display as ipd
import math, random
import torch
import torchaudio
from torchaudio import transforms
import matplotlib.pyplot as plt
import numpy as np

class AudioProcessing():

    @staticmethod
    def load(file):
        """
        Load an audio file and return:
         - the audio signal as a tensor with shape (channels, time)
         - the sample rate
        """
        return torchaudio.load(file)
    
    @staticmethod
    def to_channels(audio, n_channels):
        """
        Change the number of channels of the input loaded audio to the specified number
        (convert from stereo to mono by selecting only the first channel 
        or viceversa by duplicating the first channel)
        """
        assert n_channels==1 or n_channels==2, "Unsupported number of channels: " + str(n_channels)

        sig, sr = audio

        if (sig.shape[0] == n_channels):
            return audio
        
        if (n_channels == 1):
            new_sig = sig[:1, :]
        else:
            new_sig = torch.cat([sig, sig])

        return new_sig, sr
    
    @staticmethod
    def to_sample_rate(audio, new_sr):
        """
        Change the sample rate of the input loaded audio to the specified new sample rate
        """
        sig, sr = audio
        if (sr == new_sr):
            return audio
        
        # resample each channel
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sr)
        new_sig = resampler(sig[:1,:])
        if (sig.shape[0] > 1):
            new_sig_2 = resampler(sig[1:,:])
            new_sig = torch.cat([new_sig, new_sig_2])

        return new_sig, new_sr
    
    @staticmethod
    def to_length(audio, ms_len):
        """
        Change the length of the input loaded audio to the new length specified in milliseconds
        by padding or truncating the original signal
        """
        sig, sr = audio
        n_chan, sig_len = sig.shape
        new_len = sr//1000 * ms_len

        if (sig_len > new_len):
            sig = sig[:, :new_len]

        elif (sig_len < new_len):

            # pad with silence at the beginning and end of the signal
            pad_begin_len = random.randint(0, new_len - sig_len)
            pad_end_len = new_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((n_chan, pad_begin_len))
            pad_end = torch.zeros((n_chan, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return sig, sr

    @staticmethod
    def time_shift(audio, shift_lim):
        """
        Perform data augmentation on raw sound signal by shifting it to the left or right
        by a random amount within a limit percentage
        """

        sig, sr = audio
        _, sig_len = sig.shape
        shift_amt = int(random.uniform(-shift_lim, shift_lim) * sig_len)

        shifted_sig = torch.roll(sig, shifts=shift_amt, dims=1)
        return shifted_sig, sr
    
    @staticmethod
    def to_log_mel_spectogram(audio, n_mels=64, n_fft=1024, hop_len=None, top_db=80):
        """
        Convert a loaded audio signal to a log mel spectogram, 
        with decibel scale instead of amplitude (log mel spectogram)

        Params:
            n_mels: number of Mel bands (more bands give finer frequency resolution)
            n_fft: size of the Fast Fourier Transform window
            hop_len: hop length between frames (if None defaults to n_fft//4)
            top_db: max dynamic range in decibels (anything below max - top_dp is clamped)

        Returns:
            a log mel spectogram as a tensor of shape (channels, n_mels, time)
        """
        sig, sr = audio
        mel_spec = transforms.MelSpectrogram(sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_len) (sig)
        log_mel_spec = transforms.AmplitudeToDB(top_db=top_db) (mel_spec)

        return log_mel_spec
    
    @staticmethod
    def plot_mel_spectogram(spec):
        """
        Visualize a mel spectogram as a color map indicating decibel values for each Mel band and time frame
        """
        n_chan = spec.shape[0]
        assert n_chan == 1 or n_chan == 2, "Incorrect number of channels: " + str(n_chan)

        if n_chan == 1:
            mel_spec = spec.squeeze(0).numpy()

            plt.figure(figsize=(10, 4))
            plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
            plt.title("Log-Mel Spectrogram")
            plt.xlabel("Time frames")
            plt.ylabel("Mel bands")
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.show()

        else:
            # if dealing with stereo audio, visualize both channels
            mel_spec_L = spec[0,:,:].numpy()
            mel_spec_R = spec[1,:,:].numpy()

            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            axs[0].imshow(mel_spec_L, aspect='auto', origin='lower', cmap='magma')
            axs[0].set_title("Left Channel Log-Mel Spectrogram")

            axs[1].imshow(mel_spec_R, aspect='auto', origin='lower', cmap='magma')
            axs[1].set_title("Right Channel Log-Mel Spectrogram")

            plt.xlabel("Time frames")
            plt.tight_layout()
            plt.show()

    @staticmethod
    def time_freq_mask(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        """
        Perform data augmentation on a mel spectogram 
        by randomly masking out ranges of consecutive frequencies (add horizontal bars) or time frames (add vertical bars)
        (masked sections are replaced with the mean value)

        Params:
            spec: spectrogram tensor of shape (channels, n_mels, time)
            max_mask_pct: max portion to mask (as a fraction of size)
            n_freq_masks: number of frequency masks to apply
            n_time_masks: number of time masks to apply

        Returns:
            augmented spectrogram
        """
        _, n_mels, n_frames = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        max_freq_mask = max_mask_pct * n_mels
        max_time_mask = max_mask_pct * n_frames

        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(max_freq_mask) (aug_spec, mask_value)

        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(max_time_mask) (aug_spec, mask_value) 

        return aug_spec    
