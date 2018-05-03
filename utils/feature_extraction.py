import wave
import numpy as np
import pydub
from python_speech_features import mfcc, logfbank, delta


def audio_features(file_path, win_len=0.04, win_step=0.02):
    # MFCC and Filterbank
    wav = wave.open(file_path, 'r')
    nchannels, sampwidth, framerate, nframes, comptype, compname = wav.getparams()
    sig = wav.readframes(nframes)
    sig = np.fromstring(sig, dtype=np.int16)
    # Set nfft to be large
    nfft = 2048
    # MFCC features
    mfcc_feat = mfcc(sig, framerate, winlen=win_len, winstep=win_step, nfft=nfft)
    mfcc_delta = delta(mfcc_feat, 1)
    # Fbank features
    fbank_feat = logfbank(sig, framerate, winlen=win_len, winstep=win_step, nfft=nfft)
    fbank_delta = delta(fbank_feat, 1)
    mfcc_concat = np.concatenate((mfcc_feat, mfcc_delta),axis=0)
    fbank_concat = np.concatenate((fbank_feat, fbank_delta),axis=0)
    return mfcc_concat, fbank_concat


def convert_mp3_to_wav(mp3_file):
    """
    Function for converting mp3 file format to wav
    """
    sound = pydub.AudioSegment.from_mp3(mp3_file)
    filename = mp3_file.split('.mp3')[0] + '.wav'
    sound.export(filename, format="wav")

