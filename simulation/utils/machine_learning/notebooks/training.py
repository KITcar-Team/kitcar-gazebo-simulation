#In[]:
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
trainer.train(dataset_name='hls_default_road',epochs=1,max_inputs=0,validation_factor=0.1)

#%%
test_con = dbcon.SegmentationDatabaseConnector(mask_folder=None)
#%%
test_imgs = db_connector.load_all(dataset_name='hls_default_road',random=True,max_count=10)
predictions = trainer.predict_tests(test_imgs)
#%%
predictions[0][0][1]
#%%
idx = 0
for i in range(0,len(test_imgs) -1):
    plt.subplot(221)
    plt.imshow(test_imgs[i].load_input_img(size=[320,160]))
    plt.title('Image')
    plt.subplot(222)
    plt.imshow(test_imgs[i].load_mask_tensor(size=[320,160])[idx])
    plt.show()
    plt.imshow(predictions[i][idx].detach())
    plt.title('Result')
    plt.show()

#%%
import os
test_con.insert_folder(os.path.join(os.environ.get("KITCAR_REPO_PATH"),'kitcar-simulation-data','segmented','test'),delete=True)

#%%
os.path.join(os.environ.get("KITCAR_REPO_PATH"),'kitcar-simulation-data','segmented','test')

#%%
db_connector.connection.close()

#%%
import img.transformer as tf

#%%
img,mask=tf.rand_trafo(test_imgs[0].load_input_img(size=[320,160]),test_imgs[0].load_mask_img(size=[320,160]))

#%%
plt.subplot(221)
plt.imshow(img)
plt.title('Image')
plt.subplot(222)
plt.imshow(test_imgs[0].load_input_img(size=[320,160]))
plt.title('Result')
plt.show()

#%%
