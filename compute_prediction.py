import argparse
from os import lseek
import sys
import json
import scipy.io
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

from tensorflow.python.autograph.core import converter 
sys.path.insert(0,'./utils')
import data_preparation as dp
import padding_functions as pf 
from Dataset import DataSet
import numpy as np
import DBLSTM as mfcc2ppg
import DBLSTM_PPG_MCEP as ppgs2mcep

config_file = "config.json"


config_ppg_mcep = {}
config_mfcc_mcep = {}
config_mfcc2ppg = {}


def set_global_variables(file_name):
  with open(file_name) as json_file:
    _dict = json.load(json_file)
    global config_ppg_mcep
    config_ppg_mcep = _dict["PPG2MCEP"]
    global config_mfcc_mcep
    config_mfcc_mcep = _dict['MCEP']
    global config_mfcc2ppg
    config_mfcc2ppg = _dict['MFCC2PPG']
    # print(config_net['dim_ppgs'])
    global input_sentence
    #PATH_TO_DATA = _dict["PATH_TO_DATA_F"] # f speaker
    input_sentence = _dict["input_sentence"] # m speaker
    global target_scaler
    target_scaler = _dict["target_scaler"]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('config_file', type=str, nargs=1,
                      help='File with training Configuration')
  args = parser.parse_args()
  config_file = args.config_file[0]
  set_global_variables(config_file)

  # print(config_net)
  dictpath = "label_dict_"+str(config_ppg_mcep['dim_ppgs'])+".json"
  inverse_dictpath = "label_inv_dict_"+str(config_ppg_mcep['dim_ppgs'])+".json"
  
  # load files (list of tuples (mfcc, mcep))
  mfcc = dp.load_mfcc(input_sentence, config_mfcc_mcep)
  # load converter mfcc to ppgs
  converter = mfcc2ppg.DBLSTM( batch_size=config_mfcc2ppg['batch_size'], 
                              sequence_length=config_mfcc2ppg['sequence_length'], 
                              n_mffc=config_mfcc2ppg["n_mfcc"],
                              hidden_units=config_mfcc2ppg["hidden_units"],
                              out_classes=config_mfcc2ppg["dim_ppgs"])
  converter.load_weights(config_mfcc2ppg['path'])
  
  # transform mfcc to ppgs with model
  print('Dio')
  ppg_sentence = converter.predict(mfcc)

  target_scaler = dp.load_json_dict(target_scaler)
  target_scaler['mean'] = np.array(target_scaler['mean'])
  target_scaler['std'] = np.array(target_scaler['std'])
  # create network
  transformer = ppgs2mcep.DBLSTM(dim_ppgs=config_ppg_mcep["dim_ppgs"], dim_mceps=config_ppg_mcep["dim_mceps"], hidden_units=config_ppg_mcep["hidden_units"], 
                          batch_size=config_ppg_mcep["batch_size"], 
                          scaler=target_scaler)
                        
  transformer.load_model(config_ppg_mcep['path'])

  
  result = transformer.predict(ppg_sentence)
  scipy.io.savemat("../resconv/result"+".mat",{"mcep": result.numpy()})
    # split train test
    # sort and padd
    # train
