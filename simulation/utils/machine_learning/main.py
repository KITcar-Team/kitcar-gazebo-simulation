#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
import numpy as np
import importlib
import os
import nn.classifier
import nn.unet as unet
import database.dataset as ds
from img import transformer
import color_classes as cc
    
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from nn.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback
from nn.test_callbacks import PredictionsSaverCallback
import helpers

import database.connector as dbcon


# In[3]:


importlib.reload(nn.unet)
importlib.reload(cc)
importlib.reload(transformer)
importlib.reload(ds)
importlib.reload(nn.classifier)
importlib.reload(nn.train_callbacks)
importlib.reload(nn.test_callbacks)
importlib.reload(dbcon)

# In[8]:


# Hyperparameters
img_resize = [320, 160]
batch_size = 2
epochs = 5
threshold = 0.5
validation_size = 0.2
sample_size = None  # Put None to work on full dataset


# In[11]:


net = unet.UNet1024((3, 160,320))
classifier = nn.classifier.CarvanaClassifier(net)


# In[12]:


db_connector = dbcon.SegmentationDatabaseConnector()


# In[13]:


db_images = db_connector.load_all('hls_default_road',random=True,max_count=0)
file_count = len(db_images)
train_ds = ds.SegmentationImageDataset(segmentation_images=db_images[int(file_count*validation_size):])
valid_ds = ds.SegmentationImageDataset(segmentation_images=db_images[:int(file_count*validation_size)])


# In[14]:


from multiprocessing import cpu_count
threads = cpu_count()
use_cuda=False


# In[15]:


train_loader = DataLoader(train_ds, batch_size,
                              sampler=RandomSampler(train_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)


# In[16]:


valid_loader = DataLoader(valid_ds, batch_size,
                              sampler=SequentialSampler(valid_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)


# In[17]:


print("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))


# In[18]:


data_folder = os.path.join(os.environ.get('KITCAR_REPO_PATH'), 'kitcar-simulation-data')
tb_viz_cb = TensorboardVisualizerCallback(os.path.join(data_folder, 'logs/tb_viz'))
tb_logs_cb = TensorboardLoggerCallback(os.path.join(data_folder, 'logs/tb_logs'))
model_saver_cb = ModelSaverCallback(os.path.join(data_folder, 'output/models/model_' +
                                                     helpers.get_model_timestamp()), verbose=True)
# Testing callbacks
pred_saver_cb = PredictionsSaverCallback(os.path.join(data_folder, 'output/submit.csv.gz'),
                                             [1280,650], threshold)

# In[]:
classifier.restore_last_model()
# In[20]:


classifier.train(train_loader, valid_loader, epochs,
                     callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])


# In[5]:


test_con = dbcon.SegmentationDatabaseConnector('segmentation_test_data',mask_folder=None)
test_imgs = test_con.load_all()


# In[19]:


test_ds = ds.SegmentationImageDataset(segmentation_images=test_imgs)
test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

# In[]:
test_images = test_ds.__getitem__(0)
# In[3]:
test_seg_img = test_ds.__getitem__(0)
#ex_tensor = test_seg_img.get_input_tensor(size=[320,160])
t = test_ds.__getitem__(0).unsqueeze(0)
res = net(t)[0][1].detach()
plt.subplot(221)
plt.imshow(test_seg_img[2])
plt.title('Image')
plt.subplot(222)
#plt.imshow(ex_label_img)
#plt.title('Label')
#plt.show()

plt.imshow(res)
plt.title('Result')
plt.show()


# In[3]:


data_folder = os.environ.get('KITCAR_REPO_PATH') + '/kitcar-simulation-data/segmented/'
train_folder = data_folder + 'input'
test_folder = data_folder + 'test'
labels_folder = data_folder + 'mask'


# In[4]:


db_connector.insert_folder(train_folder, mask_folder=labels_folder,dataset_name='hls_default_road')


# In[5]:


test_folder


# In[ ]:



