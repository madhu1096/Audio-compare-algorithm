import logging
import os
from collections import defaultdict
from pathlib import Path
from random import choice
import librosa
import numpy as np
from python_speech_features import fbank
from tqdm import tqdm
import keras.backend as K

logger = logging.getLogger(__name__)

ALPHA = 0.1
SAMPLE_RATE = 20000
TRAIN_TEST_RATIO = 0.8
CHECKPOINTS_SOFTMAX_DIR = 'checkpoints-softmax'
CHECKPOINTS_TRIPLET_DIR = 'checkpoints-triplets'
BATCH_SIZE = 32 * 3 
NUM_FRAMES = 160
NUM_FBANKS = 64

def deep_speaker_loss(y_true, y_pred, alpha=ALPHA):

    split = K.shape(y_pred)[0] // 3
    anchor = y_pred[0:split]
    positive_ex = y_pred[split:2 * split]
    negative_ex = y_pred[2 * split:]
    sap = batch_cosine_similarity(anchor, positive_ex)
    san = batch_cosine_similarity(anchor, negative_ex)
    loss = K.maximum(san - sap + alpha, 0.0)
    total_loss = K.mean(loss)
    return total_loss

def sample_from_mfcc(mfcc, max_length):
    if mfcc.shape[0] >= max_length:
        r = choice(range(0, len(mfcc) - max_length + 1))
        s = mfcc[r:r + max_length]
    else:
        s = pad_mfcc(mfcc, max_length)
    return np.expand_dims(s, axis=-1)

def read_mfcc(input_filename, sample_rate):
    audio = read(input_filename, sample_rate)
    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    audio_voice_only = audio[offsets[0]:offsets[-1]]
    mfcc = mfcc_fbank(audio_voice_only, sample_rate)
    return mfcc

def read(filename, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
    assert sr == sample_rate
    return audio


def mfcc_fbank(signal: np.array, sample_rate: int):  # 1D signal array.
    # Returns MFCC with shape (num_frames, n_filters, 3).
    filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=NUM_FBANKS)
    frames_features = normalize_frames(filter_banks)
    return np.array(frames_features, dtype=np.float32)  # Float32 precision is enough here.

def normalize_frames(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]

def pad_mfcc(mfcc, max_length):  # num_frames, nfilt=64.
    if len(mfcc) < max_length:
        mfcc = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
    return mfcc

