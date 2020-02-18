import os
import math
from image_folder import make_dataset

numImgs = 6

dir_name = 'testHe_2'

dir_A1 = os.path.join('/data1/victorleee/5k_resized/' + dir_name)
dir_A2 = os.path.join('/data1/victorleee/5k_resized/' + dir_name + '_segmap')

dir_A1_paths = make_dataset(dir_A1)
dir_A2_paths = make_dataset(dir_A2)

dir_A1_paths = sorted(dir_A1_paths)
dir_A2_paths = sorted(dir_A2_paths)

A1_size = len(dir_A1_paths)
A2_size = len(dir_A2_paths)


cmd = ''
for i in range(0, numImgs):   

    # Make Name
    A1_path = dir_A1_paths[(i % A1_size)]
    A2_path = dir_A2_paths[(i % A2_size)]    

    print(A1_path + ' ' + A2_path)


    A_name = A1_path.split(dir_name + '/',1)[1]    

    A_name3 = A_name.replace('.JPG','.png')
    A_name3 = A_name3.replace('.jpg','.png')
    A_name3 = A_name3.replace('.png','.png')
    #print(A_name)
    #print(A_name3)

    
    #print(k)
    

    part1_cmd = ' CUDA_VISIBLE_DEVICES=7 python inference.py '+ A1_path +' ' + A2_path + ' /data1/victorleee/5k_resized/' + dir_name + '_segmap_crf/' + A_name3 + ' &&'
    #print(part1_cmd)


    cmd = cmd + part1_cmd
    


cmd = cmd[1:len(cmd)-1]
print(cmd)
os.system(cmd)



