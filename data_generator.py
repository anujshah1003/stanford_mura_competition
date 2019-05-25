import os

import numpy as np
import cv2
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.utils import to_categorical

#train_df = pd.read_csv(os.path.join(base_dir,train_file),usecols=['image_path','class_name','class_label','body_part','study_type'])
#val_df = pd.read_csv(os.path.join(base_dir,val_file),usecols=['image_path','class_name','class_label','body_part','study_type'])

class DataGenerator(object):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    def __init__(self,data_path='train_data_files.csv',target_size=(224,224),num_classes=2):
        self.df = pd.read_csv(data_path,usecols=['image_path','class_label'])
        self.data = self.df.values.tolist()
        self.total_samples = len(self.data)
        self.num_classes = num_classes
        self.target_size = target_size
    
    def shuffle_dataset(self, samples):
        samples = shuffle(samples,random_state=2)
        return samples
    
    
    def generate_data(self,batch_size=10,shuffle_data=True):
        """
        Yields the next training batch.
        Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
        """ 

        num_samples = len(self.data)
        if shuffle_data:
            self.data = self.shuffle_dataset(self.data) 

        while True:   
            for offset in range(0, num_samples, batch_size):
#                print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                batch_samples = self.data[offset:offset+batch_size]
                # Initialise data_x and data_y arrays for this batch
                data_x = []
                data_y = []
                # For each example
                for batch_sample in batch_samples:
                    # Load image (X)
                    img_name = batch_sample[0]
                    label = batch_sample[1]
                    try:
                        img = cv2.imread(img_name)
                        x = cv2.resize(img,self.target_size)
                        x = x/255.
#                        img = preprocess(img)
                        
                        data_x.append(x)
                        y = to_categorical(label,self.num_classes)
                        data_y.append(y)
                        
                    except Exception as e:
                        print ('the error: ',e)
                        print ('error in the file: ',x)
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(data_x)
#                X_train = np.rollaxis(X_train,1,4)
                y_train = np.array(data_y)
        
                # The generator-y part: yield the next training batch            
                yield X_train, y_train
                
if __name__ == "__main__":
    
    base_dir = 'MURA-v1.1'
    train_file = 'train_data_files.csv'
    val_file = 'val_data_files.csv'

    class_names = {1:'positive',0:'negative'}
    xrays_cat = os.listdir(os.path.join(base_dir,'train'))
    
    train_generator = DataGenerator(data_path=os.path.join(base_dir,train_file))
    train_data_gen = train_generator.generate_data(batch_size=10,shuffle_data=True)
    total_samples = train_generator.total_samples
    print ('total samples: ',total_samples)
    
    for k in range(10):
        x,y = next(train_data_gen)
        print ('the label shape: ',y.shape)
        print ('x shape: ',x.shape)
    
    # Reinitializes the generator object
#    train_generator = data_generator(train_data,batch_size=6,shuffle=True)
    # Fit model using generator