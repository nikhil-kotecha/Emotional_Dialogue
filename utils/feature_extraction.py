import wave
import numpy as np
from scipy import signal
from python_speech_features import mfcc, logfbank, delta


def audio_features(file_path, win_len=0.02, win_step=0.01):
    # MFCC and Filterbank
    wav = wave.open(file_path, 'r')
    nchannels, sampwidth, framerate, nframes, comptype, compname = wav.getparams()
    sig = wav.readframes(nframes)
    sig = np.fromstring(sig, dtype=np.int16)
    mfcc_feat = mfcc(sig, framerate, winlen=win_len, winstep=win_step)
    mfcc_delta = delta(mfcc_feat, 1)
    fbank_feat = logfbank(sig, framerate, winlen=win_len, winstep=win_step)
    fbank_delta = delta(fbank_feat, 1)
    mfcc_concat = np.concatenate((mfcc_feat, mfcc_delta),axis=0)
    fbank_concat = np.concatenate((fbank_feat, fbank_delta),axis=0)
    return mfcc_concat, fbank_concat