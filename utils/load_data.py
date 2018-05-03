import os
import re
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

# IEMOCAP data is split into 5 sessions
# We will combine data from all sessions
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

# List of possible emotions, we will remove 'xxx' which is unknown
emotions = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad', 'xxx']

def get_sentence_labels():
    """
    Get emotion labels for each sentence ID

    :return: Dictionary of labels including emotion, valence, and duration of utterance
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

    :return: dictionary of sentence transcripts
    """
    transcripts = defaultdict(str)
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    for session in sessions:
        transcripts_path = path + session + '/dialog/transcriptions/'
        for txtfile in os.listdir(transcripts_path):
            if txtfile.endswith('.txt'):
                transcript_file = open(transcripts_path + txtfile, 'r')
                lines = transcript_file.readlines()

                sentence_ids = [line.split(']:')[0][:-2].lstrip().split(' ')[0]
                                for line in lines]
                sentences = [line.split(']:') for line in lines]

                assert(len(sentence_ids) == len(sentences))

                for i in range(len(sentence_ids)):
                    transcripts[sentence_ids[i]] = sentences[i]
    return transcripts


def trim_and_pad_transcript(sequence, max_length=100):
    """
    Trim transcripts to max_length characters
    Pad if transcript is not long enough

    :param sequence: input sequence of characters
    :param max_length: pad character sequence to max_length
    :return: trimmed or padded sequence
    """
    # Insert start of sentence symbol at beginning of sequence
    sequence.insert(0, '<sos>')
    orig_seq_length = len(sequence)

    if orig_seq_length >= max_length:
        sequence = sequence[:max_length-1]
        sequence.append('<eos>')
    else:
        pad = max_length - orig_seq_length - 1
        for _ in range(pad):
            sequence.append('<pad>')
        sequence.append('<eos>')
    return sequence, orig_seq_length + 1


def trim_and_pad_audio_data(sequence, max_length=2000):
    """
    Make audio sequences the same length
    :return: Trimmed and padded audio to fit max length
    """
    seq_length = sequence.shape[0]
    features = sequence.shape[1]
    if seq_length > max_length:
        return sequence[:max_length, :], max_length
    else:
        pad = np.zeros(shape=(max_length-seq_length, features))
        return np.concatenate((sequence, pad), axis=0), seq_length


def load_sentences(use_fbank=True,
                   max_sequence_length=2000,
                   win_len=0.02, win_step=0.01):
    """
    Loads audio for both scripted and improv sentences
    :return: dictionary with sentence ID keys
    """

    emotions_labels = get_sentence_labels()
    transcripts = get_sentence_transcripts()

    sentences = []
    orig_seq_length = []
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
                        mfcc, fbank = audio_features(session_path + '/' + sentence,
                                                     win_len, win_step)
                        if use_fbank:
                            audio_data = fbank
                        else:
                            audio_data = mfcc

                        audio_data, seq_length = trim_and_pad_audio_data(audio_data,
                                                                         max_length=max_sequence_length)
                        sentence_id = sentence.split('.')[0]

                        if emotions_labels[sentence_id]['emotion'] != 'xxx':
                            dialog_id = '_'.join(sentence_id.split('_')[:2])
                            transcript = transcripts[sentence_id]

                            orig_seq_length.append(seq_length)
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
    return sentences, orig_seq_length, labels, sentences_metadata



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


def preprocess_cv_data(audio_files, use_fbank=True):
    """
    Loads audio features for each wave file
    Loads text transcription for each audio clip

    :param audio_files: list of files to process
    :param use_fbank: uses fbank features if true, otherwise uses mfcc features
    :return: Stacked audio feature data, list of original audio lengths
    """
    audio_path = common_voice_path + 'cv-valid-train/'
    data = []
    lengths = []
    for file in audio_files:
        filename = file.split('.')[0]
        wav_filename = filename + '.wav'
        # Extract audio features from wav file
        mfcc, fbank = audio_features(audio_path + wav_filename)
        if use_fbank:
            audio_data = fbank
        else:
            audio_data = mfcc
        # Trim and pad to max length
        trimmed_data, seq_lengths = trim_and_pad_audio_data(audio_data,
                                                            max_length=1500)
        # Append to data list
        data.append(trimmed_data)
        lengths.append(seq_lengths)
    return np.stack(data), lengths


def char_to_int(char_sequence):
    """
    Convert list of characters to integers
    :param char_sequence: Sequence of characters
    :return: Sequence of integers to be used in decoding
    """
    chars = ['<sos>', '<eos>', '<pad>'] + \
            list('abcdefghijklmnopqrstuvwxyz ') \
            + list('\'')
    int_sequence = []
    for char in char_sequence:
        int_sequence.append(chars.index(char))
    return np.array(int_sequence)


def int_to_char(int_sequence):
    """
    Convert list of integers to corresponding characters
    :param int_sequence: Sequence of integers
    :return: Sequence of characters
    """
    chars = ['<sos>', '<eos>', '<pad>'] + \
            list('abcdefghijklmnopqrstuvwxyz ') \
            + list('\'')
    char_sequence = []
    for i in int_sequence:
        char_sequence.append(chars[i])
    return char_sequence


def load_common_voice_data(use_fbank=True, batch_size=1, load_test=False):
    """
    Generator for Common Voice dataset of utterances and transcripts

    :param use_fbank: Transform audio data using fbanks features, otherwise mfcc
    :param batch_size: number of data points to return
    :param load_test: If true yield from test set which is set to 20% of original data
    :yield: tuple of character labels, audio data
    """
    audio_path = common_voice_path + 'cv-valid-train/'
    labels_path = common_voice_path + 'cv-valid-train.csv'

    with open(labels_path, 'r') as f:
        lines = f.readlines()[1:]
    characters = [[char for char in line.split(',')[1]]
                  for line in lines]
    utterances_and_lengths = [trim_and_pad_transcript(utterance, max_length=100)
                               for utterance in characters]

    audio_files = os.listdir(audio_path)
    num_files = len(audio_files)
    test_size = int(0.2 * num_files)

    # Yield from test set
    if load_test:
        for idx in range(0, test_size, batch_size):
            test_files = audio_files[idx:min(idx + batch_size, test_size)]
            audio_data, audio_lengths = preprocess_cv_data(test_files, use_fbank=True)
            characters = utterances_and_lengths[idx:min(idx+batch_size, test_size)]
            utterances = [char_to_int(item[0]) for item in characters]
            char_lengths = [item[1] for item in characters]
            yield (np.stack(utterances), char_lengths, audio_data, audio_lengths)

    # Else yield from training set
    else:
        for idx in range(test_size, num_files, batch_size):
            train_files = audio_files[idx:min(idx + batch_size, num_files)]
            audio_data, audio_lengths = preprocess_cv_data(train_files, use_fbank=True)
            characters = utterances_and_lengths[idx:min(idx+batch_size, num_files)]
            utterances = [char_to_int(item[0]) for item in characters]
            char_lengths = [item[1] for item in characters]
            yield (np.stack(utterances), char_lengths, audio_data, audio_lengths)

