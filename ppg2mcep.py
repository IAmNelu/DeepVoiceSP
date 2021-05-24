import argparse
from os import lseek
import sys
import json

from tensorflow.python.autograph.core import converter 
sys.path.insert(0,'./utils')
import data_preparation as dp
import padding_functions as pf 
from Dataset import DataSet
import numpy as np
import DBLSTM as mfcc2ppg
import DBLSTM_PPG_MCEP as ppgs2mcep

config_file = "config.json"


config_net = {}
config_mfcc_mcep = {}
config_mfcc2ppg = {}


def set_global_variables(file_name):
  with open(file_name) as json_file:
    _dict = json.load(json_file)
    global config_net
    config_net = _dict['NETWORK_PARAMS']
    global config_mfcc_mcep
    config_mfcc_mcep = _dict['MCEP']
    global config_mfcc2ppg
    config_mfcc2ppg = _dict['MFCC2PPG']
    # print(config_net['dim_ppgs'])
    global PATH_TO_DATA
    #PATH_TO_DATA = _dict["PATH_TO_DATA_F"] # f speaker
    PATH_TO_DATA = _dict["PATH_TO_DATA_M"] # m speaker

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('config_file', type=str, nargs=1,
                      help='File with training Configuration')
  args = parser.parse_args()
  config_file = args.config_file[0]
  set_global_variables(config_file)

  # print(config_net)
  dictpath = "label_dict_"+str(config_net['dim_ppgs'])+".json"
  inverse_dictpath = "label_inv_dict_"+str(config_net['dim_ppgs'])+".json"
  
  # load files (list of tuples (mfcc, mcep))
  mfcc_mcep, target_scaler = dp.load_mfcc_mceps(PATH_TO_DATA, config_mfcc_mcep)
  # load converter mfcc to ppgs
  converter = mfcc2ppg.DBLSTM( batch_size=config_mfcc2ppg['batch_size'], 
                              sequence_length=config_mfcc2ppg['sequence_length'], 
                              n_mffc=config_mfcc2ppg["n_mfcc"],
                              hidden_units=config_mfcc2ppg["hidden_units"],
                              out_classes=config_mfcc2ppg["dim_ppgs"])
  converter.load_weights(config_mfcc2ppg['path'])
  
  # transform mfcc to ppgs with model
  print('Dio')
  X, labels = dp.get_ppgs_mceps(converter, mfcc_mcep)

  # create network
  net2 = ppgs2mcep.DBLSTM(dim_ppgs=config_net["dim_ppgs"], dim_mceps=config_net["dim_mceps"], hidden_units=config_net["hidden_units"], 
                          batch_size=config_net["batch_size"], lr=config_net["lr"], epochs=config_net["epochs"], 
                          dropout=config_net["dropout"], decay_rate=config_net["lr_decay"], decay_steps=config_net["decay_steps"],
                          checkpoint_path=config_net["check_points"], best_checkpoint_path=config_net["best_checkpoint"], 
                          last_checkpoint_path=config_net["final_checkpoint"], log_path=config_net["log_path"], scaler=target_scaler)
  # train network
  net2.train_model(X, labels)
    # split train test
    # sort and padd
    # train
