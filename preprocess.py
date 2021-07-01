"""
1- load a file
2- pad the signal (if necessary)
3- extracting log spectrogram
4- normalize spectrogram
5- save the normalized spectrogram

PreprocessingPipeline
"""
import os
import pickle

import librosa
import numpy as np


class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        # print("inside Loader.load")
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        # print(signal.shape)
        return signal


class Padder:
    """Padder is responsible to apply padding to an array."""

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracs log spectrograms (in dB) from
    time-series signal."""

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        # (1 + frame_size / 2, num_frames)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormalizer:
    """MinMaxNormalizer applies min max normalization to an array."""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_max
        return array


class Saver:
    """saver is responsible to save features, and the min max values."""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,
                                 "min_max_values.pkl")
        print(save_path)
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory , applying
        1- load a file
        2- pad the signal (if necessary)
        3- extracting log spectrogram
        4- normalize spectrogram
        5- save the normalized spectrogram
    storing the min max values for al the log spectrograms.
    """

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.save = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                # print(file)
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                # print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        # print("inside _process_file")
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            # print("necessary")
            signal = self._apply_padding(signal)
        # print(f"test {signal.shape}")
        # print(signal)
        feature = self.extractor.extract(signal)
        # print(feature.shape)
        norm_feature = self.normalizer.normalize(feature)
        # print("after normalizer")
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            # print(self._num_expected_samples)
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74
    SAMPLE_RATE = 22050
    MONO = True

    # SPECTROGRARMS_SAVE_DIR = "/home/tozeki/datasets/fsdd/spectrograms/"
    # MIN_MAX_VALUES_SAVE_DIR = "/home/tozeki/datasets/fsdd"
    # FILES_DIR = "/home/tozeki/datasets/fsdd/audio/"

    SPECTROGRARMS_SAVE_DIR = "../datasets/fsdd/spectrograms/"
    MIN_MAX_VALUES_SAVE_DIR = "../datasets/fsdd/"
    FILES_DIR = "../datasets/fsdd/audio/"

    # instantiate all objects

    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normalizer = MinMaxNormalizer(0, 1)
    saver = Saver(SPECTROGRARMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipline = PreprocessingPipeline()
    preprocessing_pipline.loader = loader
    preprocessing_pipline.padder = padder
    preprocessing_pipline.extractor = log_spectrogram_extractor
    preprocessing_pipline.normalizer = min_max_normalizer
    preprocessing_pipline.saver = saver

    preprocessing_pipline.process(FILES_DIR)

