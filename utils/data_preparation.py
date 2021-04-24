import os
import json
import librosa
import numpy as np
import pandas as pd
import pickle


def paths_from_region(path_to_data):
    regions = os.listdir(path_to_data)
    _paths = []
    for region in regions:
        speakers = os.listdir(path_to_data + region + "/")
        for speaker in speakers:
            speaker_path = path_to_data + region + "/" + speaker
            speaches_path_full = os.listdir(speaker_path)
            names = [speaker_path + "/" +
                     f for f in set([file.split(".")[0] for file in speaches_path_full])]
            _paths += names
    return _paths


def load_json_dict(file_name):
    with open(file_name) as json_file:
        foldings_dict = json.load(json_file)
    return foldings_dict


def save_json_dict(data, file_name):
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)


def substitute_phonemes(file,
                        sentences=False,  # should we apply !ENTER/!EXIT for start/end?
                        foldings={},  # substitute phones with foldings dict
                        startend_sil=False):  # should we substitute start and end w/ sil
    fullname = file
    # print(fullname)
    phones_before = []
    phones_after = []
    os.rename(fullname, fullname+'~')
    fr = open(fullname+'~', 'r')
    fw = open(fullname, 'w')
    text_buffer = []
    all_lines = fr.readlines()
    # print(all_lines)
    for line in all_lines:
        phones_before.append(line.split()[-1])  # phone last elt of line
        tmpline = line
        tmpline = tmpline.replace('-', '')
        tmp = tmpline.split()
        for k, v in foldings.items():
            if tmp[-1] == k:
                tmp[-1] = v
                tmpline = ' '.join(tmp)
        text_buffer.append(tmpline.split())
    first_phone = text_buffer[0][-1].strip()
    last_phone = text_buffer[-1][-1].strip()
    if sentences:
        if first_phone == 'h#' or first_phone == 'sil' or first_phone == '<s>' or first_phone == '{B_TRANS}':
            # 'h#' or 'sil' for TIMIT
            # '<s>' for CSJ (and other XML/Thomas-like)
            # '{B_TRANS}' for Buckeye
            text_buffer[0] = text_buffer[0][:-1] + ['!ENTER']
        if last_phone == 'h#' or last_phone == 'sil' or last_phone == '</s>' or last_phone == '{E_TRANS}':
            text_buffer[-1] = text_buffer[-1][:-1] + ['!EXIT']
    if startend_sil:
        text_buffer[0] = text_buffer[0][:-1] + ['sil']
        text_buffer[-1] = text_buffer[-1][:-1] + ['sil']
    for buffer_line in text_buffer:
        phones_after.append(buffer_line[-1])
        fw.write(' '.join(buffer_line) + '\n')
    fw.close()
    fr.close()
    os.remove(fullname+'~')


def get_phonemes(file_path):
    with open(file_path, "r") as f:
        all_lines = f.readlines()
        phns = set([l.split(' ')[-1].strip() for l in all_lines])
        return phns


def normalize_mfcc(mfcc):
    """Normalize mfcc data using the following formula:
    normalized = (mfcc - mean)/standard deviation
    Args:
      mfcc (numpy.ndarray):
        An ndarray containing mfcc data.
        Its shape is [sentence_length, coefficients]
    Returns:
      numpy.ndarray:
        An ndarray containing normalized mfcc data with the same shape as
        the input.
    """
    means = np.mean(mfcc, 0)
    stds = np.std(mfcc, 0)
    return (mfcc - means)/stds


def compute_mfcc(paths, config_mfcc):
    _data_x = {}
    for p in paths:
        x, _ = librosa.load(p + ".WAV", sr=config_mfcc["sampling_frequency"])
        mfccs = librosa.feature.mfcc(y=x,
                                     sr=config_mfcc["sampling_frequency"],
                                     n_mfcc=config_mfcc["order_mfcc"],
                                     n_fft=config_mfcc["n_fft"],
                                     hop_length=config_mfcc["hop_length"])
        mfccs = normalize_mfcc(mfccs)
        id_ = p.split("/")[-2] + "_" + p.split("/")[-1]

        _data_x[id_] = (mfccs, p)
    return _data_x


def read_phn(f, temp_mfcc, phonem_dict, phoneme_wise=False):
    # Read PHN files
    temp_phones = pd.read_csv(f, delimiter=" ", header=None,  names=[
                              'start', 'end', 'phone'])

    if phoneme_wise:
        phones_list = []
        mfcc_block_list = []

    # Get the length of the phone data
    _, phn_len, _ = temp_phones.iloc[-1]
    phn_len_mill = int(phn_len/160)  # 160 since each frame is 10ms
    if not phoneme_wise:
        if phn_len_mill < temp_mfcc.shape[1]:
            # An array of class labels for every 10 ms in the phone data
            # phones[2] is the phoneme annotated from 20-30 ms
            phones = np.zeros(
                (len(set(phonem_dict.values())), phn_len_mill), dtype=int)
            # Make sure the length of mfcc_data and phn_len_mill are equal
            mfcc_data = temp_mfcc[:, 0:phn_len_mill]
        else:
            phones = np.zeros((set(len(phonem_dict.values())), temp_mfcc.shape[1]))
            mfcc_data = temp_mfcc

    d = phn_len_mill - temp_mfcc.shape[1]

 # Convert the string phonemes to class labels
    for i, (s, e, phone) in enumerate(temp_phones.iloc):
        start = int(s/160.0)
        end = int(e/160.0)
        if phoneme_wise:
            one_hot = np.zeros(len(set(phonem_dict.values())))
            one_hot[phonem_dict[phone]] = 1
            phones_list.append(one_hot)
            mfcc_block_list.append(
                temp_mfcc[:, start: min(end, temp_mfcc.shape[1])])
        else:
            # print(f"{start} s, {end} e, {len}")
            phones[phonem_dict[phone], start:min(end, phones.shape[1])] = 1
        # print(f"{phone} found at index {ALL_PHONEMES.index(phone)}, y becomes {phones[:,start:min(end,phones.shape[1])]}")

    if phoneme_wise:
        return np.array(phones_list), mfcc_block_list, d

    return phones.astype(int), mfcc_data, d


# receives an entry of the dictionary with key user_sentenceID -> (mfcc, path)
def match_data(sentence_entry, phonem_dict, verbose=False, phoneme_wise=False):
    """Match label to mfcc
      Args:
        sentence_entry (Tuple):
          A tuple of two elements, (mfccs, path): mfcc:np array shape paths
      Returns:
        Tuple:
          Mfccs and label paierd
    """
    phoneme_file = sentence_entry[1]+".PHN"

    mfcc_data = sentence_entry[0]

    phones, mfcc_data, d = read_phn(
        phoneme_file, mfcc_data, phonem_dict, phoneme_wise=phoneme_wise)
    if verbose:
        if d != 0:
            if abs(d) > 500:
                print(f"length mismatch of {d} frames {sentence_entry[-1]}")
    return mfcc_data, phones


def pair_data(x_dictionay, phonem_dict, phoneme_wise=False):
    result_dict = {}
    for k, v in x_dictionay.items():
        mfcc, y = match_data(v, phonem_dict, verbose=True,
                             phoneme_wise=phoneme_wise)
        result_dict[k] = {"mfcc": mfcc.T, "y": y.T, "path": v[-1]}
    return result_dict


def save_dict(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)


def load_dict(path):
    with open(path, 'rb') as f:
        loaded_obj = pickle.load(f)
        return loaded_obj
