import os
import re
import csv
import string
from collections import defaultdict, namedtuple

import numpy as np
import pydub

from .feature_extraction import audio_features

#################################################################
# Emotion Recognition Load Data
# IEMOCAP dataset
#################################################################

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


def load_sentences(use_fbank=True, max_sequence_length=2000):
    """
    Loads audio for both scripted and improv sentences
    :return: dictionary with sentence ID keys
    """

    emotions_labels = get_sentence_labels()
    transcripts = get_sentence_transcripts()

    sentences = []
    sentences_metadata = []
    labels = []
    Audio = namedtuple('Audio', ['id',
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

                        audio_data = trim_and_pad_audio_data(audio_data)
                        sentence_id = sentence.split('.')[0]

                        if emotions_labels[sentence_id]['emotion'] != 'xxx':
                            dialog_id = '_'.join(sentence_id.split('_')[:2])
                            transcript = transcripts[sentence_id]

                            sentences.append(audio_data)
                            labels.append(emotions_labels[sentence_id]['emotion'])

                            sentences_metadata.append(Audio(sentence_id,
                                                      emotions_labels[sentence_id]['duration'],
                                                      emotions_labels[sentence_id]['emotion'],
                                                      transcript,
                                                      dialog_id,
                                                      dataset))
        print('Finished data session ' + session)
    sentences = np.stack(sentences, axis=0)
    return sentences, labels, sentences_metadata



#################################################################
# Speech Recognition Load Data
# Common Voice Dataset from Mozilla
#################################################################

common_voice_path = os.path.abspath(os.curdir) + '/cv_corpus_v1/'

def mp3_to_wav(set):
    """
    Convert mp3 to wav
    """
    audio_path = common_voice_path + 'cv-valid-{}/'.format(set)
    audio_files = os.listdir(audio_path)

    num_files = len(audio_files)
    for i, file in enumerate(audio_files):
        if i % 10000 == 0:
            print('Processing {} of {} files in {}'.format(i, num_files, set))
        filename = file.split('.')[0]
        wav_filename = filename + '.wav'
        if wav_filename not in audio_files:
            # Convert mp3 file and save as .wav file
            sound = pydub.AudioSegment.from_mp3(audio_path +  file)
            # Delete mp3 file to save space
            os.remove(audio_path + file)
            # Save sound as wave file in path
            sound.export(audio_path + wav_filename, format="wav")


def preprocess_common_voice_data(set, use_fbank=True):
    """
    Loads audio features for each wave file
    Loads text transcription for each audio clip

    :param set: 'dev', 'train', 'test'
    :param use_fbank: uses fbank features if true, otherwise uses mfcc features
    :return: List of audio feature data, List of transcription labels
    """
    labels_path = common_voice_path + 'cv-valid-{}.csv'.format(set)
    labels = []
    with open(labels_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            characters = list(line[1])
            characters.insert('<sos>', 0)
            characters.append('<eos>')
            labels.append(characters)

    audio_path = common_voice_path + 'cv-valid-{}/'.format(set)
    data = []
    audio_files = os.listdir(audio_path)
    for file in audio_files:
        filename = file.split('.')[0]
        wav_filename = filename + '.wav'
        if wav_filename not in audio_files:
            # Convert mp3 file and save as .wav file
            sound = pydub.AudioSegment.from_mp3(audio_path +  file)
            # Delete mp3 file to save space
            os.remove(audio_path + file)
            # Save sound as wave file in path
            sound.export(audio_path + wav_filename, format="wav")
        # Extract audio features from wav file
        mfcc, fbank = audio_features(audio_path + wav_filename)
        if use_fbank:
            audio_data = fbank
        else:
            audio_data = mfcc
        # Append to data list
        data.append(audio_data)
    return data, labels


def load_common_voice_data(use_fbank=True, include_test=False):
    """

    :param use_fbank:
    :param include_test:
    :return:
    """
    # get data and labels by train, dev, test
    train_data, train_labels = preprocess_common_voice_data('train', use_fbank)
    dev_data, dev_labels = preprocess_common_voice_data('dev', use_fbank)
    if include_test:
        test_data, test_labels = preprocess_common_voice_data('test', use_fbank)
        return train_data, train_labels, dev_data, dev_labels, test_data, test_labels
    else:
        return train_data, train_labels, dev_data, dev_labels


