import os

import numpy as np
import cv2
import pandas as pd
import random

base_dir = 'MURA-v1.1'
train_image_file = 'train_image_paths.csv'
train_label_file = 'train_labeled_studies.csv'
val_image_file = 'valid_image_paths.csv'
val_label_file = 'valid_labeled_studies.csv'


train_img_df = pd.read_csv(os.path.join(base_dir,train_image_file),header=None)
val_img_df = pd.read_csv(os.path.join(base_dir,val_image_file),header=None)
train_df = pd.read_csv(os.path.join(base_dir,train_label_file),header=None)
val_df = pd.read_csv(os.path.join(base_dir,val_label_file),header=None)

train_img_df.columns = ['image_path']
val_img_df.columns = ['image_path']

#def assign_labels(df):
#    labels = []
#    for i in range(len(df)):
#        fname = df['image_path'][i]
#        if 'positive' in fname:
#            label = 1
#        elif 'negative' in fname:
#            label=0
#        print(fname[-19:-11])
#        labels.append(label)
#    return labels
#    
#train_labels,tr_class_names = assign_labels(train_img_df)
#val_labels,val_class_names = assign_labels(val_img_df)

train_img_df['class_name']=train_img_df['image_path'].apply(lambda x: str(x.split('/')[4])[7:])
train_img_df['class_label']=train_img_df['class_name'].apply(lambda x: 0 if x=='negative' else 1 )
train_img_df['body_part']=train_img_df['image_path'].apply(lambda x: str(x.split('/')[2])[3:])
train_img_df['study_type']=train_img_df['image_path'].apply(lambda x: str(x.split('/')[4])[:6])


val_img_df['class_name']=val_img_df['image_path'].apply(lambda x: str(x.split('/')[4])[7:])
val_img_df['class_label']=val_img_df['class_name'].apply(lambda x: 0 if x=='negative' else 1 )
val_img_df['body_part']=val_img_df['image_path'].apply(lambda x: str(x.split('/')[2])[3:])
val_img_df['study_type']=val_img_df['image_path'].apply(lambda x: str(x.split('/')[4])[:6])



train_img_df.to_csv('MURA-V1.1/train_data_files.csv',index=None)
val_img_df.to_csv('MURA-V1.1/val_data_files.csv',index=None)

