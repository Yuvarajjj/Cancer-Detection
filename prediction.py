import pandas as pd
import numpy as np
import os
import tensorflow
import shutil
import glob
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
train_path  = "D:/vgg16/train - Copy"
#valid_path  = ""
test_path   = "D:/vgg16/test - Copy"

imgs = ['N','AB','X']



train_class = []
test_class  = []
for i in os.listdir(train_path):
    train_class+=[i]
print("The Clases in train path are ",train_class)
for i in os.listdir(test_path):
    test_class+=[i]
print("The Class in the test path are",test_class)



train_data_gen = ImageDataGenerator(preprocessing_function= vgg16.preprocess_input , zoom_range= 0.2, horizontal_flip= True, shear_range= 0.2 , rescale= 1./255)
train = train_data_gen.flow_from_directory(directory= train_path , target_size=(224,224))
test_data_gen = ImageDataGenerator(preprocessing_function= vgg16.preprocess_input, rescale= 1./255 )
test = train_data_gen.flow_from_directory(directory= test_path , target_size=(224,224), shuffle= False)

print(train.class_indices)

class_type = {0:'Abnormal', 1:'Normal'}
t_img , label = train.next()

def plotImages(img_arr, label):

  for im, l in zip(img_arr,label) :
    plt.figure(figsize= (5,5))
    plt.imshow(im, cmap = 'gray')
    plt.title(im.shape)
    plt.axis = True
    plt.show()
#plotImages(t_img, label)


vgg = VGG16( input_shape=(224,224,3), include_top= False) # include_top will consider the new weights

for layer in vgg.layers:           # Dont Train the parameters again
  layer.trainable = False
x = Flatten()(vgg.output)
x = Dense(units=3, activation='softmax', name = 'predictions')(x)

model = Model(vgg.input, x)
model.summary()
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor= "val_accuracy" , min_delta= 0.01, patience= 3, verbose=1)
mc = ModelCheckpoint(filepath="D:/PROJECT-VGG/bestmodel_30.h5", monitor="val_accuracy", verbose=1, save_best_only= True)

#hist = model.fit_generator(train, steps_per_epoch= 10, epochs= 8, validation_data= valid , validation_steps= 32)

#hist = model.fit(train, steps_per_epoch= 2, epochs= 8 , validation_data= test , validation_steps= 32, callbacks=[mc])
hist = model.fit(train, epochs= 1, validation_data= test, callbacks=[mc])
## load only the best model


model = load_model("D:/PROJECT-VGG/bestmodel_30.h5")

h = hist.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'] , c = "red")
plt.title("acc vs v-acc")
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'] , c = "red")
plt.title("loss vs v-loss")
plt.show()

acc = model.evaluate_generator(generator= test)[1]

print(f"The accuracy of your model is = {acc} %")


def get_img_array(img_path):
  """
  Input : Takes in image path as input
  Output : Gives out Pre-Processed image
  """
  path = img_path

  img = load_img(path, target_size=(224, 224, 3))

  img = keras.utils.img_to_array(img) / 255
  img = np.expand_dims(img,   axis = 0)

  return img

#  PREDICTION
# path for that new image. ( you can take it either from google or any other scource)

path = "D:/vgg16/test - Copy/Normal/N 3.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()



path = "D:/vgg16/test - Copy/Normal/N 3.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()

path = "D:/vgg16/test - Copy/Blind Sample Images for Test/X2.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()

path = "D:/vgg16/test - Copy/Blind Sample Images for Test/X8.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()

path = "D:/vgg16/Blind Sample Images for Test/X6.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()

path = "D:/vgg16/Blind Sample Images for Test/X5.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()

path = "D:/vgg16/Blind Sample Images for Test/X4.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()

path = "D:/vgg16/Blind Sample Images for Test/X3.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()

path = "D:/vgg16/Blind Sample Images for Test/X2.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()

path = "D:/vgg16/Blind Sample Images for Test/X1.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()

path = "D:/vgg16/Blind Sample Images for Test/X9.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()

path = "D:/vgg16/Blind Sample Images for Test/X10.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given  image is of type = {res}")
print()

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()
