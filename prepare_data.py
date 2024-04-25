from __future__ import division
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob

# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare ISIC 2018 data set #################################################
Dataset_add = 'Data/'
Tr_add = 'Train/'

Tr_list = glob.glob(Dataset_add + Tr_add+'/*.jpg')
# It contains 18998 training samples
Data_train    = np.zeros([18998, height, width, channels])
Label_train   = np.zeros([18998, height, width])

## TODO adapt this code to my problem 

print('Reading Data ISIC')
for idx in range(len(Tr_list)):
    print(idx+1)
    img = sc.imread(Tr_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_train[idx, :,:,:] = img

    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ 'ISIC2018_Task1_Training_GroundTruth/' + b +'_segmentation.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train[idx, :,:] = img2    
         
print('Reading ISIC  finished')

################################################################ Make the train and test sets ########################################    
# We consider 1815 samples for training, 259 samples for validation and 520 samples for testing

Train_img      = Data_train[0:1815,:,:,:]
Validation_img = Data_train[1815:1815+259,:,:,:]
Test_img       = Data_train[1815+259:2594,:,:,:]

Train_mask      = Label_train[0:1815,:,:]
Validation_mask = Label_train[1815:1815+259,:,:]
Test_mask       = Label_train[1815+259:2594,:,:]


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)

