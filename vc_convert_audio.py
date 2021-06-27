import argparse
from os import lseek
import sys
import json
import scipy.io
import os
import pyworld as pw
import librosa
import pysptk
import scipy.io.wavfile 



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
    global PATH_TO_DATA
    PATH_TO_DATA = _dict["PATH_TO_DATA"] # VCTK dataset
    global input_sentence
    input_sentence = _dict["input_sentence"] # m speaker
    global SPEAKER
    SPEAKER = _dict['SPEAKER']
    global target_scaler
    target_scaler = _dict["target_scaler"] + SPEAKER + "scaler.json"
    global final_directory
    final_directory = _dict['RESULT_FOLDER']

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
  
  # load converter mfcc to ppgs
  converter = mfcc2ppg.DBLSTM( batch_size=config_mfcc2ppg['batch_size'], 
                              sequence_length=config_mfcc2ppg['sequence_length'], 
                              n_mffc=config_mfcc2ppg["n_mfcc"],
                              hidden_units=config_mfcc2ppg["hidden_units"],
                              out_classes=config_mfcc2ppg["dim_ppgs"])
  converter.load_weights(config_mfcc2ppg['path'])
  
  # transform mfcc to ppgs with model
  # print('Dio')

  target_scaler = dp.load_json_dict(target_scaler)
  target_scaler['mean'] = np.array(target_scaler['mean'])
  target_scaler['std'] = np.array(target_scaler['std'])
  # create network
  transformer = ppgs2mcep.DBLSTM(dim_ppgs=config_ppg_mcep["dim_ppgs"], dim_mceps=config_ppg_mcep["dim_mceps"], hidden_units=config_ppg_mcep["hidden_units"], 
                          batch_size=config_ppg_mcep["batch_size"], 
                          scaler=target_scaler)
                        
  transformer.load_model(config_ppg_mcep['path']+SPEAKER+"_ppg2mcep.ckpt")

  with open(input_sentence, 'r') as ft:
    lines = ft.readlines()
    for l in lines:
      l = l.strip()
      target_s, source_s, file_s = l.split('_')
      if target_s != SPEAKER:
        continue
      filename = PATH_TO_DATA + source_s + "/" + source_s + "_" + file_s
      mfcc = dp.load_mfcc(filename, config_mfcc_mcep)
      ppg_sentence = converter.predict(mfcc)
      result = transformer.predict(ppg_sentence).numpy()
      
      #extract info from source file
      x, _ = librosa.load(filename, sr=config_mfcc_mcep["sampling_frequency"])
      x = x.astype(np.float64)
      _f0, t = pw.dio(x, config_mfcc_mcep["sampling_frequency"])# frame_period=10)
      f0_try = pw.stonemask(x, _f0, t, config_mfcc_mcep["sampling_frequency"]) #refinement of f0 using stone mask
      ap_try = pw.d4c(x=x, f0=_f0, temporal_positions=t, fs=config_mfcc_mcep["sampling_frequency"],fft_size=config_mfcc_mcep["n_fft"])

      #use transformed result
      indices = sorted(np.concatenate([np.arange(len(result))]*2))
      alpha=0.35
      spc = pysptk.mc2sp(result[indices], alpha, config_mfcc_mcep["n_fft"]).astype(np.float64)[:len(ap_try)]
      y2 = pw.synthesize(f0_try, spc, ap_try, config_mfcc_mcep["sampling_frequency"])
      endfile = final_directory+l
      scipy.io.wavfile.write(endfile, config_mfcc_mcep["sampling_frequency"], y2)
      
