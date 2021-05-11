import argparse
import sys
import json 
sys.path.insert(0,'./utils')
import data_preparation as dp
import padding_functions as pf 
from Dataset import DataSet
import numpy as np
# import DBLSTM as net

config_file = "config.json"

config_run = {}  # set what to run and what not to
config_net = {}

def set_global_variables(file_name):
  with open(file_name) as json_file:
    _dict = json.load(json_file)
    config_net = _dict['NETWORK_PARAMS']
    print(config_net)
    
    

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('config_file', type=str, nargs=1,
                      help='File with training Configuration')
  args = parser.parse_args()
  config_file = args.config_file[0]
  set_global_variables(config_file)

  # dictpath = "label_dict_"+str(NUM_CLASSES)+".json"
  # inverse_dictpath = "label_inv_dict_"+str(NUM_CLASSES)+".json"