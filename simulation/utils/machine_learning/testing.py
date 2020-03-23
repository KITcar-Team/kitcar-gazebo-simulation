#In[]:

import sys,os
sys.path.insert(1, os.path.join(os.environ['KITCAR_REPO_PATH'],'kitcar-gazebo-utils','machine-learning'))

from nn.trainer import Trainer,DatabaseTrainer
import nn.unet as unet
import nn.classifier
from matplotlib import pyplot as plt
from nn.fast_scnn import FastSCNN

import database.connector as dbcon
#%
net = unet.UNet128((1, 160,320))
fast_scnn = FastSCNN(7)
classifier = nn.classifier.CarvanaClassifier(net)

db_connector = dbcon.SegmentationDatabaseConnector()


trainer = DatabaseTrainer(net,db_connector,classifier)

#%%
trainer.restore_last_model()

#%%
test_con = dbcon.SegmentationDatabaseConnector(mask_folder=None)
#%%
test_imgs = db_connector.load_all(dataset_name='hls_default_road',random=True,max_count=10)
predictions = trainer.predict_tests(test_imgs)

#%%
idx =6
for i in range(0,len(test_imgs) -1):
    plt.subplot(221)
    plt.imshow(test_imgs[i].load_input_img(size=[320,160]))
    plt.title('Image')
    plt.subplot(222)
    plt.imshow(test_imgs[i].load_mask_tensor(size=[320,160])[idx])
    plt.subplot(223)
    plt.imshow(predictions[i][idx].detach())
    plt.title('Result')
    plt.subplot(224)
    plt.imshow(cv2.threshold(cv2.UMat(predictions[i][idx].detach().numpy()),0.5,1,cv2.THRESH_BINARY)[1].get())
    plt.show()#%%


#%%
import cv2

#%%
cv2.threshold(cv2.UMat(predictions[1][idx].detach().numpy()),0.5,1,cv2.THRESH_BINARY)[1]

#%%
predictions[1][2].detach().numpy()
#%%
cv2.threshold(cv2.UMat(predictions[1][2].detach().numpy()),0.5,1,cv2.THRESH_BINARY)[1].get()

#%%
