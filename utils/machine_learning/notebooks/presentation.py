#%% Imports
import sys,os
sys.path.insert(1, os.path.join(os.environ['KITCAR_REPO_PATH'],'kitcar-gazebo-utils','machine-learning'))

import cv2
import img.transformer as transformer
import color_classes

from matplotlib import pyplot as plt

#%% File path
FILE = os.path.join(os.environ.get('KITCAR_REPO_PATH'), 'kitcar-simulation-data','segmented/mask/','presentation.png')
mask = cv2.imread(FILE)
#%%
tensor = transformer.mask_to_class_tensor(mask)
tensor.shape
#%%
idx = color_classes.ColorClass.MIDDLE_LINE.value

plt.subplot(121)
plt.imshow(mask)
plt.subplot(122)
plt.imshow(tensor[idx])
plt.show()#%%

#%%
