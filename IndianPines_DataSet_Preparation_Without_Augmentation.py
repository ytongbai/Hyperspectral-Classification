
# coding: utf-8

# Import dependencies
# ===================

# In[1]:

import scipy.io
import numpy as np
from random import shuffle
import random
import scipy.ndimage
from skimage.transform import rotate
import os
import patch_size

# Load dataset
# ===========

# In[2]:

DATA_PATH = os.path.join(os.getcwd(),"Data")
input_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines.mat'))['indian_pines']
target_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_gt.mat'))['indian_pines_gt']

# Define global variables
# =======================

# In[3]:

HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
PATCH_SIZE = patch_size.patch_size
TRAIN_PATCH,TRAIN_LABELS,TEST_PATCH,TEST_LABELS = [],[],[],[]
CLASSES = [] 
COUNT = 200 #Number of patches of each class
OUTPUT_CLASSES = 16
TEST_FRAC = 0.8 #Fraction of data to be used for testing

# In[4]:

PATCH_SIZE

# Scale the input between [0,1]
# ==========================

# In[5]:

input_mat = input_mat.astype(float)
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)

# Calculate the mean of each channel for normalization
# ====================================================

# In[6]:

MEAN_ARRAY = np.ndarray(shape=(BAND,),dtype=float)
for i in range(BAND):
    MEAN_ARRAY[i] = np.mean(input_mat[:,:,i])

# In[7]:

def Patch(height_index,width_index):
    """
    Returns a mean-normalized patch, the top left corner of which 
    is at (height_index, width_index)
    
    Inputs: 
    height_index - row index of the top left corner of the image patch
    width_index - column index of the top left corner of the image patch
    
    Outputs:
    mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE) 
    whose top left corner is at (height_index, width_index)
    """
    transpose_array = np.transpose(input_mat,(2,0,1))
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i]) 
    
    return np.array(mean_normalized_patch)

# Collect all available patches of each class from the given image
# ================================================================

# In[8]:

for i in range(OUTPUT_CLASSES):
    CLASSES.append([])
for i in range(HEIGHT - PATCH_SIZE + 1):
    for j in range(WIDTH - PATCH_SIZE + 1):
        curr_inp = Patch(i,j)
        curr_tar = target_mat[i + int((PATCH_SIZE - 1)/2), j + int((PATCH_SIZE - 1)/2)]
        if(curr_tar!=0): #Ignore patches with unknown landcover type for the central pixel
            CLASSES[curr_tar-1].append(curr_inp)

# In[9]:

for c  in CLASSES:
    print len(c)

# Make a test split with 25% data from each class
# ===============================================

# In[10]:

for c in range(OUTPUT_CLASSES): #for each class
    class_population = len(CLASSES[c])
    test_split_size = int(class_population*TEST_FRAC)
        
    patches_of_current_class = CLASSES[c]
    shuffle(patches_of_current_class)
    
    #Make training and test splits
    TRAIN_PATCH.append(patches_of_current_class[:-test_split_size])
        
    TEST_PATCH.extend(patches_of_current_class[-test_split_size:])
    TEST_LABELS.extend(np.full(test_split_size, c, dtype=int))

# In[11]:

for c in TRAIN_PATCH:
    print len(c)


# Oversample the classes which do not have at least COUNT patches in the training set and extract COUNT patches
# =============================================================================================================

# In[12]:

for i in range(OUTPUT_CLASSES):
    if(len(TRAIN_PATCH[i])<COUNT):
        tmp = TRAIN_PATCH[i]
        for j in range(COUNT/len(TRAIN_PATCH[i])):
            shuffle(TRAIN_PATCH[i])
            TRAIN_PATCH[i] = TRAIN_PATCH[i] + tmp
    shuffle(TRAIN_PATCH[i])
    TRAIN_PATCH[i] = TRAIN_PATCH[i][:COUNT]
    


# In[13]:

for c in TRAIN_PATCH:
    print len(c)


# In[14]:

TRAIN_PATCH = np.asarray(TRAIN_PATCH)

# In[15]:

TRAIN_PATCH = TRAIN_PATCH.reshape((-1,220,PATCH_SIZE,PATCH_SIZE))

# In[16]:

TRAIN_LABELS = np.array([])
for l in range(OUTPUT_CLASSES):
    TRAIN_LABELS = np.append(TRAIN_LABELS, np.full(COUNT, l, dtype=int))

# Augment the data with random flipped and rotated patches
# ========================================================

# In[1]:

# for i in range(OUTPUT_CLASSES):
#     shuffle(CLASSES[i])
#     for j in range(COUNT/2): #There will be COUNT/2 original patches and COUNT/2 randomly rotated/flipped patches of each class
#         num = random.randint(0,2)
#         if num == 0 :
#             flipped_patch = np.flipud(CLASSES[i][j]) #Flip patch up-down
#         if num == 1 :
#             flipped_patch = np.fliplr(CLASSES[i][j]) #Flip patch left-right
#         if num == 2 :
#             no = random.randrange(-180,180,30)
#             flipped_patch = scipy.ndimage.interpolation.rotate(CLASSES[i][j], no,axes=(1, 0), 
#                     reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=False) #Rotate patch by a random angle
#         TRAIN_PATCH.append(CLASSES[i][j])
#         TRAIN_LABELS.append(i)
#         TRAIN_PATCH.append(flipped_patch)
#         TRAIN_LABELS.append(i)

#     for j in range(COUNT/2,COUNT/2 + 100):
#         num = random.randint(0,2)
#         if num == 0 :
#             flipped_patch = np.flipud(CLASSES[i][j])
#         if num == 1 :
#             flipped_patch = np.fliplr(CLASSES[i][j])
#         if num == 2 :
#             no = random.randrange(-180,180,30)
#             flipped_patch = scipy.ndimage.interpolation.rotate(CLASSES[i][j], no, axes=(1, 0), reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=False)
#         TEST_PATCH.append(CLASSES[i][j])
#         TEST_LABELS.append(i)
#         TEST_PATCH.append(flipped_patch)
#         TEST_LABELS.append(i)

# In[2]:

print len(TEST_PATCH)
print len(TRAIN_PATCH)

# Save the patches in segments
# =================================

# 1. Training data
# ----------------

# In[6]:

for i in range(len(TRAIN_PATCH)/(COUNT*2)):
    train_dict = {}
    start = i * (COUNT*2)
    end = (i+1) * (COUNT*2)
    file_name = 'Train_'+str(PATCH_SIZE)+'_'+str(i+1)+'.mat'
    train_dict["train_patch"] = TRAIN_PATCH[start:end]
    train_dict["train_labels"] = TRAIN_LABELS[start:end]
    scipy.io.savemat(os.path.join(DATA_PATH, file_name),train_dict)
    print i,

# 2. Test data
# -------------

# In[4]:

for i in range(len(TEST_PATCH)/(COUNT*2)):
    test_dict = {}
    start = i * (COUNT*2)
    end = (i+1) * (COUNT*2)
    file_name = 'Test_'+str(PATCH_SIZE)+'_'+str(i+1)+'.mat'
    test_dict["test_patch"] = TEST_PATCH[start:end]
    test_dict["test_labels"] = TEST_LABELS[start:end]
    scipy.io.savemat(os.path.join(DATA_PATH, file_name),test_dict)

# In[5]:

len(TRAIN_PATCH)/(COUNT*2)


# In[6]:



# In[6]:



# In[6]:



# In[6]:



# In[6]:



# In[ ]:


