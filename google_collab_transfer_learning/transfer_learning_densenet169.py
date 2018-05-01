import numpy as np
import keras
from keras import optimizers
from keras import applications
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD
import pickle

fileObject = open('x_train.pickle','rb')  
# load the object from the file into var b
x_train = pickle.load(fileObject)  

fileObject2 = open('y_train.pickle','rb')  
# load the object from the file into var b
y_train = pickle.load(fileObject2)

fileObject3 = open('x_test.pickle','rb')  
# load the object from the file into var b
x_test = pickle.load(fileObject3)

fileObject4 = open('y_test.pickle','rb')  
# load the object from the file into var b
y_test = pickle.load(fileObject4)

x_train = np.stack(x_train)
y_train = np.stack(y_train)
x_test = np.stack(x_test)
y_test = np.stack(y_test)


y_train = keras.utils.to_categorical(y_train, num_classes=22)

y_test = keras.utils.to_categorical(y_test, num_classes=22)


model = applications.densenet.DenseNet169(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))


for layer in model.layers[:-3]:
    layer.trainable = False
    
#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(22, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.fit(x_train, y_train, validation_split=0.33, epochs=5, batch_size=10)
score = model_final.evaluate(x_test, y_test,  verbose=0,)
print("%s: %.2f%%" % (model_final.metrics_name))