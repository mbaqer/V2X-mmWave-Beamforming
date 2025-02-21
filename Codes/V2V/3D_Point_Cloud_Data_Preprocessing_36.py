# -*- coding: utf-8 -*-
"""
Created on Tue August 13 14:57:57 2023

"""


import os
import numpy as np
import pandas as pd


from numpy.random import RandomState




###############################################
#### Input dataset name
###############################################
root_folder = 'scenario36'
data_csv = './scenario36/scenario36.csv'



###############################################
# Read dataset to create a list of the input sequence   
###############################################

df = pd.read_csv(data_csv)
image_data_lst = df['unit1_lidar1'].values # LiDAR
pwr_data_lst = df['unit1_pwr3'].values
original_beam_1 = df['unit1_pwr3_best-beam'].values
original_index = df['abs_index'].values  # Retrieve 'index' values from the original file


###############################################
#### subsample the power and generate the 
#### updated beam indices
###############################################
updated_beam = []
original_beam = []
for entry in pwr_data_lst:
    data_to_read = f'./{root_folder}{entry[1:]}'
    pwr_data = np.loadtxt(data_to_read)
    original_beam.append(np.argmax(pwr_data)+1)
    updated_pwr = []
    j = 0
    while j < (len(pwr_data)- 1):
        tmp_pwr = pwr_data[j]
        updated_pwr.append(tmp_pwr)
        j += 2
    updated_beam.append(np.argmax(updated_pwr)+1)
    

# For LiDAR
def create_img_beam_dataset():
    
    folder_to_save = 'Main_Folder'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    
    #############################################
    ###### created updated image path ###########
    #############################################
    updated_img_path = []
    for entry in image_data_lst:
        img_path = entry.split('./')[1]
        updated_path = f'../{root_folder}/{img_path}'
        updated_img_path.append(updated_path)
    
    
    #############################################
    # saving the image-beam development dataset for training and validation
    #############################################
                            
    # indx = np.arange(1, len(updated_beam)+1,1)
    
    original_index_name = 'original_index'  # New column name for the 'index' values
    indx = original_index  # Use the 'index' values from the original file
    df_new = pd.DataFrame()
    df_new[original_index_name] = indx
    df_new['unit1_lidar1'] = updated_img_path  # Lidar
    df_new['original_unit1_pwr3_best-beam'] = original_beam_1
    
    # Adding additional columns with original data from the input file
    df_new['original_unit1_pwr3'] = df['unit1_pwr3']
    
    df_new.to_csv(fr'./{folder_to_save}/scenario36_64_lidar_beam.csv', index=False)       
      
    #############################################
    #generate the train and test dataset
    #############################################    
    rng = RandomState(1)
    train, val, test = np.split(df_new.sample(frac=1, random_state=rng ), [int(.6*len(df_new)), int(.9*len(df_new))])
    train.to_csv(f'./{folder_to_save}/scenario36_64_lidar_beam_train.csv', index=False)
    val.to_csv(f'./{folder_to_save}/scenario36_64_lidar_beam_val.csv', index=False)
    test.to_csv(f'./{folder_to_save}/scenario36_64_lidar_beam_test.csv', index=False)    


if __name__ == "__main__":   
    create_img_beam_dataset()

    
