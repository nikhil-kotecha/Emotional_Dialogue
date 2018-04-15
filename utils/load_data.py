import os
import re
import string
import numpy as np
from collections import defaultdict, namedtuple
from .feature_extraction import audio_features

path = os.path.abspath(os.curdir) + '/IEMOCAP_full_release/'
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
emotions = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad', 'xxx']

def get_sentence_labels():
    """
    Get emotion labels for each sentence ID
    :return:
    """
    labels = defaultdict(dict)
    for session in sessions:
        labels_path = path + session + '/dialog/EmoEvaluation/'
        for txtfile in os.listdir(labels_path):
            if txtfile.endswith('.txt'):
                label_file = open(labels_path + txtfile, 'r')
                lines = label_file.readlines()
                lines = [line for line in lines if line[0] == '[']
                for line in lines:
                    time_info, sentence_id, emotion, valence = line.split('\t')
                    if emotion not in emotions:
                        emotion = 'xxx'
                    time_info = time_info[1:-1].split(' - ')
                    times = [float(time) for time in time_info]
                    labels[sentence_id] = {'emotion': emotion,
                                           'valence': valence[:-1],
                                           'start_time': times[0],
                                           'end_time': times[1],
                                           'duration': times[1] - times[0]}
    return labels


def get_sentence_transcripts():
    """
    Transcripts of sentences cleaned by removing punctuation and lower casing

    :return:
    """
    transcripts = defaultdict(str)
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    for session in sessions:
        transcripts_path = path + session + '/dialog/transcriptions/'
        for txtfile in os.listdir(transcripts_path):
            if txtfile.endswith('.txt'):
                transcript_file = open(transcripts_path + txtfile, 'r')
                lines = transcript_file.readlines()

                sentence_ids = [line.split(']:')[0][:-2].lstrip().split(' ')[0] for line in lines]
                sentences = [line.split(']:') for line in lines]

                assert(len(sentence_ids) == len(sentences))

                for i in range(len(sentence_ids)):
                    transcripts[sentence_ids[i]] = sentences[i]
    return transcripts



def load_sentences(use_fbank=True):
    """
    Loads audio for both scripted and improv sentences
    :return: dictionary with sentence ID keys
    """

    emotions_labels = get_sentence_labels()
    transcripts = get_sentence_transcripts()

    sentences = []
    Audio = namedtuple('Audio', ['id',
                                 'audio_data',
                                 'duration',
                                 'emo_label',
                                 'transcript',
                                 'dialog_id',
                                 'dataset'])
    for session in sessions:
        wav_path = path + session + '/sentences/wav'

        wav_dirs = os.listdir(wav_path)
        for dir in wav_dirs:
            if not dir.startswith('.'):
                dataset = dir.split('_')[1][:3]
                session_path = wav_path + '/' + dir
                for sentence in os.listdir(session_path):
                    if sentence.endswith('.wav'):
                        mfcc, fbank = audio_features(session_path + '/' + sentence)
                        if use_fbank:
                            audio_data = fbank
                        else:
                            audio_data = mfcc
                        sentence_id = sentence.split('.')[0]
                        dialog_id = '_'.join(sentence_id.split('_')[:2])
                        transcript = transcripts[sentence_id]
                        sentences.append(Audio(sentence_id,
                                                  audio_data,
                                                  emotions_labels[sentence_id]['duration'],
                                                  emotions_labels[sentence_id]['emotion'],
                                                  transcript,
                                                  dialog_id,
                                                  dataset))
        print('Finished data session ' + session)
    return sentences


def trim_and_pad_audio_data(sequence, max_length=2000):
    """
    Make audio sequences the same length
    :return:
    """
    seq_length = sequence.shape[0]
    features = sequence.shape[1]
    if seq_length > max_length:
        return sequence[:2000, :]
    else:
        pad = np.zeros(shape=(max_length-seq_length, features))
        return np.concatenate((sequence, pad), axis=0)



