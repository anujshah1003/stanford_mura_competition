from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

class NNModels(object):
    
    '''
    A set of different Neural Network models for training 
    '''
    def __init__(self):
        pass
    
    def base_model(self,input_shape,num_classes):
        
        model = Sequential()

        model.add(Conv2D(32, (3,3),input_shape=input_shape))
        model.add(Activation('relu'))
        
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.5))
        
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        #model.add(Convolution2D(64, 3, 3))
        #model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        
        return model
        
if __name__=='__main__':
    model_obj = NNModels()
    model = model_obj