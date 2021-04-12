import argparse
import sys
import json 
sys.path.insert(0,'./utils')
import data_preparation as dp
import padding_functions as pf 
from Dataset import DataSet
import numpy as np
import DBLSTM as net


config_file = "config.json"

config_run = {} # set what to run and what not to

def set_global_variables(file_name):
  with open(file_name) as json_file:
    _dict = json.load(json_file)
  global PATH_TO_TRAIN
  global PATH_TO_TEST
  global PATH_FOLDING
  global CHECKPOINT_PATH 
  global NUM_CLASSES
  global log_file_path
  global config_run
  global config_mfcc
  global SQ
  global BATCH_SIZE
  global EPOCHS
  global HIDDEN_UNITS
  global LR
  global DropOut
  PATH_TO_TRAIN = _dict["TRAIN_PATH"]
  PATH_TO_TEST = _dict["TEST_PATH"]
  PATH_FOLDING = _dict["FOLDING_DICT"]
  CHECKPOINT_PATH = _dict["CHECKPOINT_PATH"]
  log_file_path = _dict["LOG_FILE"]
  NUM_CLASSES = _dict["NUM_CLASSES"]
  config_run["process_data"] = _dict["propPROCESS_DATA"]
  config_run["process_phonemes"] = _dict["propUPDATE_PHONEMES"]
  config_run["propTRAIN"] = _dict["propTRAIN"]
  config_run["propLOAD"] = _dict["propLOAD"]
  config_mfcc = _dict["MFCC_DATA"]
  SQ = _dict["SQ"]
  BATCH_SIZE = _dict["BATCH_SIZE"]
  EPOCHS = _dict["EPOCHS"]
  HIDDEN_UNITS = _dict["HIDDEN_UNITS"]
  LR = _dict["LR"]
  DropOut = _dict["DROPOUT"] 

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('config_file', type=str, nargs=1, help='File with training Configuration')
  args = parser.parse_args()
  config_file = args.config_file[0]
  set_global_variables(config_file)

  if config_run["process_data"]: # process files 
    # Load all paths 
    train_paths = dp.paths_from_region(PATH_TO_TRAIN)
    test_paths = dp.paths_from_region(PATH_TO_TEST)
    print(f"Files for train: {len(train_paths)}")
    print(f"Files for test: {len(test_paths)}")

    # Compute MFCCs
    train_dict_x = dp.compute_mfcc(train_paths, config_mfcc)
    print(f"###  All data loded: {len(train_paths) == len(train_dict_x.keys())} ###")
    test_dict_x = dp.compute_mfcc(test_paths, config_mfcc)
    print(f"###  All data loded: {len(test_paths) == len(test_dict_x.keys())} ###")

    
    if config_run["process_phonemes"]: # substitute phonemes
      # load phonemes dictionary
      dict_f = dp.load_json_dict(PATH_FOLDING)
      set_phns = set()
      for p in (train_paths + test_paths):
        phn_file = p + ".PHN"
        dp.substitute_phonemes(phn_file, foldings=dict_f)
        setphn = dp.get_phonemes(phn_file)
        set_phns = set_phns.union(setphn)
      phonemes_dict = {i:phn for i, phn in enumerate(list(set_phns))}
      phonemes_dict_inv = {v:k for k,v in phonemes_dict.items()}
      
      dp.save_json_dict(phonemes_dict,'label_dict.json')
      dp.save_json_dict(phonemes_dict_inv,'label_inv_dict.json')
  
    else: # load phonemes
      phonemes_dict = dp.load_json_dict('label_dict.json')
      phonemes_dict_inv = dp.load_json_dict('label_inv_dict.json')

    train_data = dp.pair_data(train_dict_x, phonemes_dict_inv)
    dp.save_dict(train_data, 'train_data.pickle')
    
    test_data = dp.pair_data(test_dict_x, phonemes_dict_inv)
    dp.save_dict(test_data, 'test_data.pickle')

  else: # load pickles
    phonemes_dict = dp.load_json_dict('label_dict.json')
    phonemes_dict_inv = dp.load_json_dict('label_inv_dict.json')
    train_data = dp.load_dict("train_data.pickle")
    test_data = dp.load_dict("test_data.pickle")

  Xp_tr, label_tr = pf.pad_data(train_data, seq_length=SQ)
  Xp_ts, label_ts = pf.pad_data(test_data, seq_length=SQ)

  train_dataset = DataSet(np.array(Xp_tr, dtype=object), np.array(label_tr, dtype=object), BATCH_SIZE)    
  test_dataset = DataSet(np.array(Xp_ts, dtype=object), np.array(label_ts, dtype=object), BATCH_SIZE)

  print("Model Crearion")
  model = net.DBLSTM(batch_size=BATCH_SIZE, sequence_length=SQ, n_mffc=config_mfcc["order_mfcc"],
              hidden_units=HIDDEN_UNITS, out_classes=NUM_CLASSES, dropout=DropOut, num_epochs=EPOCHS, log=log_file_path,
               LR=LR, ch_path=CHECKPOINT_PATH)
  
  if config_run["propLOAD"]:
    model.load_weights(CHECKPOINT_PATH.format(epoch=0))
  print("Start Training")
  if config_run["propTRAIN"]:
    model.train_model(train_dataset, test_dataset)
