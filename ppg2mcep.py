import argparse
from os import lseek
import sys
import json 
sys.path.insert(0,'./utils')
import data_preparation as dp
import padding_functions as pf 
from Dataset import DataSet
import numpy as np
# import DBLSTM as net

config_file = "config.json"


config_net = {}
config_mfcc_mcep = {}

def set_global_variables(file_name):
  with open(file_name) as json_file:
    _dict = json.load(json_file)
    global config_net
    config_net = _dict['NETWORK_PARAMS']
    global config_mfcc_mcep
    config_mfcc_mcep = _dict['MCEP']
    # print(config_net['dim_ppgs'])
    global PATH_TO_DATA
    PATH_TO_DATA = _dict["PATH_TO_DATA_F"] # f speaker
    # PATH_TO_DATA = _dict["PATH_TO_DATA_M"] # m speaker

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
  
  mfcc_mcep = dp.load_mfcc_mceps(PATH_TO_DATA, config_mfcc_mcep)
  