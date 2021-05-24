import argparse
import sys
import json 
sys.path.insert(0,'./utils')
import data_preparation as dp
import padding_functions as pf 
from Dataset import DataSet
import numpy as np
# import DBLSTM as net
import DBLSTM_MFCC_PPG as net

config_file = "config.json"

config_run = {}  # set what to run and what not to


def set_global_variables(file_name):
  with open(file_name) as json_file:
    _dict = json.load(json_file)
  global PATH_TO_TRAIN
  PATH_TO_TRAIN = _dict["TRAIN_PATH"]
  global PATH_TO_TEST
  PATH_TO_TEST = _dict["TEST_PATH"]
  global PATH_FOLDING
  PATH_FOLDING = _dict["FOLDING_DICT"]
  global config_net
  config_net = _dict["NETWORK_PARAM"]
  global config_run
  config_run = _dict["CONFIG_RUN"]
  global config_mfcc
  config_mfcc = _dict["MFCC_DATA"]
  global PHONEME_WISE
  PHONEME_WISE = _dict["PHONEME_WISE"]
  global CHECKPOINT_PATH
  CHECKPOINT_PATH =  config_net["checkpoint_path"]
  global padding_silence
  padding_silence = _dict["PADDING_SIL"]

if __name__ == "__main__":
  #parse config file and set var
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('config_file', type=str, nargs=1,
                      help='File with training Configuration')
  args = parser.parse_args()
  config_file = args.config_file[0]
  set_global_variables(config_file)

  dictpath = "config_files/label_dict_"+str(config_net["num_phoneme_classes"])+".json"
  inverse_dictpath = "config_files/label_inv_dict_"+str(config_net["num_phoneme_classes"])+".json"

  if config_run["process_data"]:  # process files
    # Load all paths
    train_paths = dp.paths_from_region(PATH_TO_TRAIN)
    test_paths = dp.paths_from_region(PATH_TO_TEST)
    print(f"Files for train: {len(train_paths)}")
    print(f"Files for test: {len(test_paths)}")

    # Compute MFCCs
    train_dict_x = dp.compute_mfcc(train_paths, config_mfcc)
    print(
        f"###  All data loaded: {len(train_paths) == len(train_dict_x.keys())} ###")
    test_dict_x = dp.compute_mfcc(test_paths, config_mfcc)
    print(
        f"###  All data loaded: {len(test_paths) == len(test_dict_x.keys())} ###")

    if config_run["process_phonemes"]:  # substitute phonemes using the dictionary chosen
      # load phonemes dictionary
      dict_f = dp.load_json_dict(PATH_FOLDING)
      set_phns = set()
      for p in (train_paths + test_paths):
        phn_file = p + ".PHN"
        dp.substitute_phonemes(phn_file, foldings=dict_f)
        setphn = dp.get_phonemes(phn_file)
        set_phns = set_phns.union(setphn)
      #create phoneme direct/inverse dictionary 
      phonemes_dict = {i: phn for i, phn in enumerate(list(set_phns))}
      phonemes_dict_inv = {v: k for k, v in phonemes_dict.items()}

      #save phoneme dictionaries
      dp.save_json_dict(phonemes_dict, dictpath)
      dp.save_json_dict(phonemes_dict_inv, inverse_dictpath)

    else:  #don't process files, just load phoneme dictionaries
      phonemes_dict = dp.load_json_dict(dictpath)
      phonemes_dict_inv = dp.load_json_dict(inverse_dictpath)

    #pair mfcc frame to label
    train_data = dp.pair_data(
        train_dict_x, phonemes_dict_inv, PHONEME_WISE)
    dp.save_dict(train_data, 'train_data.pickle')

    test_data = dp.pair_data(test_dict_x, phonemes_dict_inv, PHONEME_WISE)
    dp.save_dict(test_data, 'test_data.pickle')

  else:  # no process, just load pickles
    phonemes_dict = dp.load_json_dict(dictpath)
    phonemes_dict_inv = dp.load_json_dict(inverse_dictpath)
    train_data = dp.load_dict("train_data.pickle")
    test_data = dp.load_dict("test_data.pickle")

  if config_run["propRMSilencePostLoad"]:
    #remove strting and ending silence
    sil_num = phonemes_dict_inv['sil']
    train_data = pf.remove_silence(train_data, sil_num, padding_silence)
    test_data = pf.remove_silence(test_data, sil_num,padding_silence)
  Xp_tr, label_tr = pf.pad_data(train_data, seq_length=config_net["sq"])
  Xp_ts, label_ts = pf.pad_data(test_data, seq_length=config_net["sq"])

  train_dataset = DataSet(np.array(Xp_tr, dtype=object), np.array(
      label_tr, dtype=object), config_net["batch_size"])
  test_dataset = DataSet(np.array(Xp_ts, dtype=object),
                          np.array(label_ts, dtype=object), config_net["batch_size"])
  print("Model Creation")
  # model = net.DBLSTM(batch_size=config_net["batch_size"], sequence_length=config_net["sq"], n_mffc=config_mfcc["order_mfcc"],
  #               hidden_units=config_net["hidden_units"], out_classes=config_net["num_phoneme_classes"], dropout=config_net["dropout"], 
  #               num_epochs=config_net["epochs"], log=config_net["log_file"],LR=config_net["lr"], 
  #               decay_rate=config_net["lr_decay"], decay_steps=config_net["decay_steps"], 
  #               ch_path=config_net["checkpoint_path"], best_path=config_net["best_checkpoint"], last_path = config_net["final_checkpoint"])

  # if config_run["propLOAD"]:
  #     model.load_weights(CHECKPOINT_PATH.format(epoch=0))
  # print("Start Training")
  # if config_run["propTRAIN"]:
  #     model.train_model(train_dataset, test_dataset)
  train_data, train_label = pf.get_data_from_dict(train_data)
  test_data, test_label = pf.get_data_from_dict(test_data)
  print(config_net)
  model = net.DBLSTM( batch_size=config_net["batch_size"], n_mffc=config_mfcc["order_mfcc"], 
                      hidden_units=config_net["hidden_units"], out_classes=config_net["num_phoneme_classes"], 
                      dropout=config_net["dropout"], num_epochs=config_net["epochs"], 
                      log=config_net["log_file"], LR=config_net["lr"], decay_rate=config_net["lr_decay"], 
                      decay_steps=config_net["decay_steps"], ch_path=config_net["checkpoint_path"],                      
                      best_path=config_net["best_checkpoint"], last_path=config_net["final_checkpoint"])
  model.train_model( train_data, train_label, test_data, test_label)