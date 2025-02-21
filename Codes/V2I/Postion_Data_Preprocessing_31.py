# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:57:57 2023
@author: Baqer
"""

import os
import numpy as np
import pandas as pd

from numpy.random import RandomState

###############################################
#### Input dataset name
###############################################
root_folder = 'scenario31'
data_csv = './scenario31/scenario31_dev.csv'

###############################################
# Read dataset to create a list of the input sequence   
###############################################
df = pd.read_csv(data_csv)
pwr_data_lst = df['unit1_pwr_60ghz'].values
original_beam_1 = df['unit1_beam'].values
original_index = df['index'].values  # Retrieve 'index' values from the original file

###############################################
#### subsample the power and generate the 
#### updated beam indices
###############################################
updated_beam = []
original_beam = []
for entry in pwr_data_lst:
    data_to_read = f'./{root_folder}{entry[1:]}'
    pwr_data = np.loadtxt(data_to_read)
  
    original_beam.append(np.argmax(pwr_data))
    updated_pwr = []
    j = 0
    while j < (len(pwr_data)- 1):
        tmp_pwr = pwr_data[j]
        updated_pwr.append(tmp_pwr)
        j += 2
    updated_beam.append(np.argmax(updated_pwr)+1)
    

def create_pos_beam_dataset():  
    folder_to_save = './Main_Folder'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    
    ###############################################
    ####### read position values from dataset #####
    ###############################################
    lat = []
    lon = []
    pos_data_path = df['unit2_loc'].values
    for entry in pos_data_path:
        data_to_read = f'./{root_folder}{entry[1:]}'
        pos_val = np.loadtxt(data_to_read)
        lat.append(pos_val[0])
        lon.append(pos_val[1])
        
    def norm_data(data_lst):
        norm_data = []
        for entry in data_lst:
            norm_data.append((entry - min(data_lst))/(max(data_lst) - min(data_lst)))
        return norm_data

    ###############################################
    ##### normalize latitude and longitude data ###
    ###############################################
    lat_norm = norm_data(lat)
    lon_norm = norm_data(lon)

    ###############################################
    ##### generate final pos data #################
    ###############################################
    pos_data = []
    for j in range(len(lat_norm)):
        pos_data.append([lat_norm[j], lon_norm[j]])

    #############################################
    # saving the pos-beam development dataset for training and validation
    #############################################
    original_index_name = 'original_index'  # New column name for the 'index' values
    indx = original_index  # Use the 'index' values from the original file
    df_new = pd.DataFrame()
    df_new[original_index_name] = indx
    df_new['unit2_gps_nor'] = pos_data   
    df_new['original_unit1_pwr_best-beam'] = original_beam_1

    # Add additional columns with original data from the input file
    df_new['original_unit1_pwr'] = df['unit1_pwr_60ghz']
    # df_new['original_unit1_gps'] = df['unit1_loc']
    # df_new['original_unit2_gps'] = df['unit2_loc']

    df_new.to_csv(fr'./{folder_to_save}/scenario31_64_pos_beam.csv', index=False) 
    
    #############################################
    #generate the train and test dataset
    #############################################    
    rng = RandomState(1)
    train, val, test = np.split(df_new.sample(frac=1, random_state=rng ), [int(.6*len(df_new)), int(.9*len(df_new))])
    train.to_csv(f'./{folder_to_save}/scenario31_64_pos_beam_train.csv', index=False)
    val.to_csv(f'./{folder_to_save}/scenario31_64_pos_beam_val.csv', index=False)
    test.to_csv(f'./{folder_to_save}/scenario31_64_pos_beam_test.csv', index=False)    

if __name__ == "__main__":   
    create_pos_beam_dataset()
