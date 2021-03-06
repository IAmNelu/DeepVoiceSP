import os
import json
import librosa
import librosa.util
import pysptk
import numpy as np
import pandas as pd
import pickle
import math

#save all paths to the data
def paths_from_region(path_to_data):
  '''
  WORKS WITH TIMIT DATASET (data organized in regions and speakers)
  takes path to dataset folder and creates a list of paths to the real data
  '''
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

#utils function to load a json dictionary
def load_json_dict(file_name):
  with open(file_name) as json_file:
    foldings_dict = json.load(json_file)
  return foldings_dict

#utils function to save a json dictionary
def save_json_dict(data, file_name):
  with open(file_name, 'w') as outfile:
    json.dump(data, outfile)


def substitute_phonemes(file, foldings={}):
  '''
  Input:
    phoneme transcription file
    dictionary to use for substitution
  Returns:
    noting, but rewrites file applying the substitutions
  '''  
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
    phones_before.append(line.split()[-1])  # phone last elem of line
    tmpline = line
    tmpline = tmpline.replace('-', '')
    tmp = tmpline.split()
    for k, v in foldings.items():
      if tmp[-1] == k: #check if phoneme has to be changed
        tmp[-1] = v
        tmpline = ' '.join(tmp)
    text_buffer.append(tmpline.split())
  #first_phone = text_buffer[0][-1].strip()
  #last_phone = text_buffer[-1][-1].strip()
  for buffer_line in text_buffer: #read all file, now rewrite it
    phones_after.append(buffer_line[-1])
    fw.write(' '.join(buffer_line) + '\n')
  fw.close()
  fr.close()
  os.remove(fullname+'~') #clean up


def get_phonemes(file_path):
  '''
  from phoneme transcripted file returns list of phonemes
  '''
  with open(file_path, "r") as f:
    all_lines = f.readlines()
    phns = set([l.split(' ')[-1].strip() for l in all_lines])
    return phns


def normalize_mfcc(mfcc):
  """Normalize mfcc data using the following formula:
  normalized = (mfcc - mean)/standard deviation
  compute mean and std for each order of mfcc for the whole sentence
  Args:
    mfcc array (size is: [sentence_length x order_mfcc])
  Returns:
    numpy.ndarray:
      An ndarray containing normalized mfcc data with the same shape as
      the input.
  """
  means = np.mean(mfcc, 0)
  stds = np.std(mfcc, 0)
  return (mfcc - means)/stds


def compute_mfcc(paths, config_mfcc):
  '''extract normalized mfcc from list of data path
  input:
    list of paths to data, 
    mfcc setting dictionary
  return:
    dictionary:
      key: speaker code + _ + audio name
      value: tuple (mfcc normalized, path)
  '''
  _data_x = {}
  for p in paths:
    if not "SA" in p:
      x, _ = librosa.load(p + ".WAV", sr=config_mfcc["sampling_frequency"])
      mfccs = librosa.feature.mfcc(y=x,
                                  sr=config_mfcc["sampling_frequency"],
                                  n_mfcc=config_mfcc["order_mfcc"],
                                  n_fft=config_mfcc["n_fft"],
                                  hop_length=config_mfcc["hop_length"])
      mfccs = normalize_mfcc(mfccs.T).T
      id_ = p.split("/")[-2] + "_" + p.split("/")[-1]

      _data_x[id_] = (mfccs, p)
  return _data_x


def read_phn(f, temp_mfcc, phonem_dict): #CHECK RATIO DONE IF WANT FRAME WITH DIFFERENT LENGTH TO 10MS
  '''perform match between mfcc extracted and corresponding phoneme
  input:
    f: phoneme transcription file path
    temp_mfcc: mfcc extracted
    phonem_dict: inverse dictionary
  returns:
    list of phones one-hot arrays, which will be the labels 
    mfcc_data: final mfcc
    d: mismatch length between mfcc extracted and phonemes in file
  '''
  # Read PHN files
  temp_phones = pd.read_csv(f, delimiter=" ", header=None,  names=[
                            'start', 'end', 'phone'])


  # Get the length of the phone data
  _, phn_len, _ = temp_phones.iloc[-1]
  phn_len_mill = int(phn_len/160)  # 160 since each frame is 10ms, CHECK IF DIFFERENT SETTINGS
  if phn_len_mill < temp_mfcc.shape[1]:
    # An array of class labels for every 10 ms in the phone data
    # phones[2] is the phoneme annotated from 20-30 ms
    phones = np.zeros((len(set(phonem_dict.values())), phn_len_mill), dtype=int)
    # Make sure the length of mfcc_data and phn_len_mill are equal
    mfcc_data = temp_mfcc[:, 0:phn_len_mill]
  else:
    phones = np.zeros((set(len(phonem_dict.values())), temp_mfcc.shape[1]))
    mfcc_data = temp_mfcc

  d = phn_len_mill - temp_mfcc.shape[1]

# Convert the string phonemes to class labels
  for i, (s, e, phone) in enumerate(temp_phones.iloc):
    start = int(s/160.0)  # 160 since each frame is 10ms, CHECK IF DIFFERENT SETTINGS
    end = int(e/160.0)  # 160 since each frame is 10ms, CHECK IF DIFFERENT SETTINGS
    
      # print(f"{start} s, {end} e, {len}")
    phones[phonem_dict[phone], start:min(end, phones.shape[1])] = 1
  # print(f"{phone} found at index {ALL_PHONEMES.index(phone)}, y becomes {phones[:,start:min(end,phones.shape[1])]}")

  return phones.astype(int), mfcc_data, d


# receives an entry of the dictionary with key user_sentenceID -> (mfcc, path)
def match_data(sentence_entry, phonem_dict, verbose=False):
  """Match label to mfcc
    Args:
      sentence_entry (Tuple):
        A tuple of two elements, (mfccs, path): mfcc:np array shape paths
      phonemes dictionary
    Returns:
        Mfccs and labels paierd
  """
  phoneme_file = sentence_entry[-1]+".PHN"

  mfcc_data = sentence_entry[0]

  phones, mfcc_data, d = read_phn(phoneme_file, mfcc_data, phonem_dict)
  if verbose:
    if d != 0:
      if abs(d) > 500:
        print(f"length mismatch of {d} frames {sentence_entry[-1]}")
  return mfcc_data, phones


def pair_data(x_dictionay, phonem_dict):
  '''
    Pair data available in dictionary of (mfcc,paths)
    returns:
      dictionary:
        key: same of input data dictionary (speaker code + _ + audio name)
        values:
          mfcc: transposed mfcc
          y: paired label
          path: path to file
  '''
  result_dict = {}
  for k, v in x_dictionay.items():
    mfcc, y = match_data(v, phonem_dict, verbose=True)
    # result_dict[k] = {"mfcc": mfcc.T, "y": y.T, "path": v[-1], "mceps":v[1]}
    result_dict[k] = {"mfcc": mfcc.T, "y": y.T, "path": v[-1]}

  return result_dict

#util function to save pickle file
def save_dict(x, path):
  with open(path, 'wb') as f:
    pickle.dump(x, f)

#util function to load pickle file
def load_dict(path):
  with open(path, 'rb') as f:
    loaded_obj = pickle.load(f)
    return loaded_obj


def load_mfcc(path_to_wav_file, config_mfcc_mceps):
  '''Extract mfcc information from a single file
  '''
  x, _ = librosa.load(path_to_wav_file, sr=config_mfcc_mceps["sampling_frequency"])
  mfccs = librosa.feature.mfcc(y=x, sr=config_mfcc_mceps["sampling_frequency"],
                                n_mfcc=config_mfcc_mceps["order_mfcc"],
                                n_fft=config_mfcc_mceps["n_fft"],
                                hop_length=config_mfcc_mceps["hop_length"])
  mfccs = normalize_mfcc(mfccs.T).T #transpose twice in order to normalize on right axis
  return mfccs.T

def load_mfcc_mceps(path_to_data, config_mfcc_mceps):
  '''extract normalized mfcc and mceps from list of data path
  input:
    list of paths to data, 
    mfcc_mceps setting dictionary
  return:
    dictionary:
      key: speaker code + _ + audio name
      value: tuple (mfcc normalized, mceps normalized)
    target scaler:
      contains mcep mean and variance of target speaker in order to scale back mcep results
  '''
  _data_x = {}
  path_audios = os.listdir(path_to_data)
  total_mceps = np.empty((0,config_mfcc_mceps['order_mcep']+1), float) #used to store mean and std for denormalize results
  target_scaler = {}
  for p in path_audios:
    if p.split(".")[-1] != "wav":
      continue
    x, _ = librosa.load(path_to_data + '/' + p, sr=config_mfcc_mceps["sampling_frequency"])

    mfcc_l = math.ceil(x.shape[0]/config_mfcc_mceps["hop_length"])
    mcep_l=math.ceil((x.shape[0]-config_mfcc_mceps["n_fft"])/config_mfcc_mceps["hop_length"] )
    final_shape = x.shape[0] + config_mfcc_mceps["hop_length"]*(mfcc_l-mcep_l)
    x.resize((final_shape,))
    frames = librosa.util.frame(x, frame_length=config_mfcc_mceps["n_fft"], hop_length=config_mfcc_mceps["hop_length"]).astype(np.float64).T
    # Windowing
    frames *= pysptk.blackman(config_mfcc_mceps["n_fft"], normalize=1)
    mceps = pysptk.mcep(frames, config_mfcc_mceps['order_mcep']) #,alpha) 
    total_mceps = np.vstack((total_mceps,mceps)) 
    mfccs = librosa.feature.mfcc(y=x, sr=config_mfcc_mceps["sampling_frequency"],
                                  n_mfcc=config_mfcc_mceps["order_mfcc"],
                                  n_fft=config_mfcc_mceps["n_fft"],
                                  hop_length=config_mfcc_mceps["hop_length"])
    mfccs = normalize_mfcc(mfccs.T).T #transpose twice in order to normalize on right axis
    id_ = "_" + p
    _data_x[id_] = (mfccs.T, mceps) #Don't forget mfcc.T -> now both have shape (#frames, #mfcc/mceps)
  
  target_scaler["mean"] = list(np.mean(total_mceps, 0))
  target_scaler["std"] = list(np.std(total_mceps, 0))

  #apply normalization 
  for k, v in _data_x.items():
    mcep = v[1]
    mcep =  (mcep - target_scaler["mean"])/target_scaler["std"]
    _data_x[k] = (v[0], mcep)

  return _data_x, target_scaler
  
def load_mfcc_mceps_VCTK(list_train_data, data_folder, speaker, config_mfcc_mceps):
  '''extract normalized mfcc and mceps from list of data path
  input:
    list_train_data: path to file name with audio tracks title for target
    data_folder: path to data folder
    speaker: code for target speaker  
    mfcc_mceps setting dictionary
  return:
    dictionary:
      key: speaker code + _ + audio name
      value: tuple (mfcc normalized, mceps normalized)
    target scaler:
      contains mcep mean and variance of target speaker in order to scale back mcep results
  '''
  root = data_folder
  _data_x = {}
  target_scaler = {}
  with open(list_train_data, 'r') as ft:
    count_errors = 0
    lines = ft.readlines()
    total_mceps = np.empty((0,config_mfcc_mceps['order_mcep']+1), float) #used to store mean and std for denormalize results
    for l in lines:
      l = l.strip()
      speaker_f, _ = l.split('_')
      if speaker_f != speaker:
        continue
      wav_path = root + speaker + '/' + l + '.wav'
      try:
        x, _ = librosa.load(wav_path, sr=config_mfcc_mceps["sampling_frequency"])
        mfccs = librosa.feature.mfcc(y=x, sr=config_mfcc_mceps["sampling_frequency"],
                                    n_mfcc=config_mfcc_mceps["order_mfcc"],
                                    n_fft=config_mfcc_mceps["n_fft"],
                                    hop_length=config_mfcc_mceps["hop_length"])
        mfccs = normalize_mfcc(mfccs.T).T #transpose twice in order to normalize on right axis
        
        ## pad the extracted x in order to frame it to have same number of mceps and mfccs
        mfcc_l = math.ceil(x.shape[0]/config_mfcc_mceps["hop_length"]) #number of 10ms frames expected
        mcep_l=math.ceil((x.shape[0]-config_mfcc_mceps["n_fft"])/config_mfcc_mceps["hop_length"] ) #number of 10ms frames without 0 padding
        final_shape = x.shape[0] + config_mfcc_mceps["hop_length"]*(mfcc_l-mcep_l) #compute new shape in order to get same number of mceps and mfcc frames
        x.resize((final_shape,))
        
        
        frames = librosa.util.frame(x, frame_length=config_mfcc_mceps["n_fft"], hop_length=config_mfcc_mceps["hop_length"]).astype(np.float64).T
        # mceps = pysptk.mcep(frames, config_mfcc_mceps['order_mcep'], etype=1, eps=1e-5)
        mceps = pysptk.mcep(frames, config_mfcc_mceps['order_mcep'])
        
        total_mceps = np.vstack((total_mceps,mceps)) 
        
        id_ = "_" + l
        _data_x[id_] = (mfccs.T, mceps) #Don't forget mfcc.T -> now both have shape (#frames, #mfcc/mceps)
      except:
        #print(f"Error file: {wav_path}")
        count_errors+=1
    
    #print(f"\nTotal errors: {count_errors}\n")
  # compute mean and std for all mceps
  target_scaler["mean"] = np.mean(total_mceps, 0)
  target_scaler["std"] = np.std(total_mceps, 0) 
  #apply normalization 
  for k, v in _data_x.items():
    mcep = v[1]
    mcep =  (mcep - target_scaler["mean"])/target_scaler["std"]
    _data_x[k] = (v[0], mcep)
  
  #convert to list to save to file
  target_scaler["mean"] = list(target_scaler["mean"])
  target_scaler["std"] = list(target_scaler["std"])
  
  print(f"Total Seconds of audio: {total_mceps.shape[0]/100}")
  
  return _data_x, target_scaler
    
    
     
def get_ppgs_mceps(converto, mfcc_mcep):
  '''
  Use pretrained phase 1 to get ppgs from mfcc
  input:
    converto: phase1 model
    mfcc_mcep exctracted dictionary
  output:
    X: computed ppgs
    y: corresponding mcep labels 
  '''
  X = []
  y = []
  for mfcc, mcep in mfcc_mcep.values():
    ppgs = converto.predict(mfcc)
    dif = ppgs.shape[0] - mcep.shape[0] # frame mismatch in some case -> mcep cut some starting silence frame
    #if dif != 0: print(dif)
    X.append(ppgs[dif:])
    y.append(mcep)
  return X, y
