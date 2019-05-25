import os
import numpy as np

from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD

from data_generator import DataGenerator
from model_file import NNModels

from pyimagesearch.resnet import ResNet
from pyimagesearch import config

img_w,img_h = 64,64
img_c = 3

input_shape = (img_w,img_h,img_c)
num_classes = 2

# define the total number of epochs to train for along with the
# initial learning rate and batch size
NUM_EPOCHS = 10
INIT_LR = 1e-1
BS = 32

def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0

	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	# return the new learning rate
	return alpha
#%%
## Define the model
#model_obj = NNModels()
#model = model_obj.base_model(input_shape,num_classes)
#
##sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
##model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
#model.summary()

# initialize our ResNet model and compile it
model = ResNet.build(64, 64, 3, num_classes, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer='adam',
	metrics=["accuracy"])
model.summary()
#%%
# define the data generator to load data
base_dir = 'MURA-v1.1'
train_file = 'train_data_files.csv'
val_file = 'val_data_files.csv'

train_generator = DataGenerator(data_path=os.path.join(base_dir,train_file),\
                                target_size=(img_w,img_h),num_classes=num_classes)
train_data_gen = train_generator.generate_data(batch_size=BS,shuffle_data=True)
num_tr_samples = train_generator.total_samples
print ('total train samples: ',num_tr_samples)

test_generator = DataGenerator(data_path=os.path.join(base_dir,val_file),\
                                target_size=(img_w,img_h),num_classes=num_classes)
test_data_gen = train_generator.generate_data(batch_size=BS,shuffle_data=False)
num_te_samples = test_generator.total_samples
print ('total test samples: ',num_te_samples)
# train the network
print("[INFO] training network...")
H = model.fit_generator(train_data_gen,
	validation_data=test_data_gen, steps_per_epoch=num_tr_samples // BS,
	epochs=NUM_EPOCHS,validation_steps=num_te_samples // BS, verbose=1)


