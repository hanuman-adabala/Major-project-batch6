import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.layers import Input, Add, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import os
import cv2

training_dir  = "data\\train"
validation_dir  = "data\\val"
testing_dir  = "data\\test"

from PIL import Image
count1=0
CATEGORIES = ['disguise', 'mask', 'scarf']
from tqdm import tqdm
for category in CATEGORIES:
      path1 = os.path.join(training_dir, category)
      path2 = os.path.join(validation_dir, category)
      class_num = CATEGORIES.index(category)
      for img in tqdm(os.listdir(path2)):
        img1 = cv2.imread(os.path.join(path1, img), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(os.path.join(path2, img), cv2.IMREAD_UNCHANGED)
        im1 = Image.open(os.path.join(path1, img))
        im2 = Image.open(os.path.join(path2, img))
        if im1 == im2:
          count1+=1
print()
print(count1,"matches found!")

from PIL import Image
count1=0
CATEGORIES = ['disguise', 'mask', 'scarf']
from tqdm import tqdm
for category in CATEGORIES:
      path1 = os.path.join(training_dir, category)
      path2 = os.path.join(testing_dir, category)
      class_num = CATEGORIES.index(category)
      for img in tqdm(os.listdir(path2)):
        img1 = cv2.imread(os.path.join(path1, img), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(os.path.join(path2, img), cv2.IMREAD_UNCHANGED)
        im1 = Image.open(os.path.join(path1, img))
        im2 = Image.open(os.path.join(path2, img))
        if im1 == im2:
          count1+=1
print()
print(count1,"matches found!")

device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:0':
  #raise SystemError('GPU device not found')
#print('Found GPU at: {}'.format(device_name))

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
valid_datagen = ImageDataGenerator()
train_data = train_datagen.flow_from_directory('tmp/train/', class_mode='categorical', target_size=(224,224), batch_size=32)
test_data = test_datagen.flow_from_directory('tmp/test/', class_mode='categorical', target_size=(224,224), batch_size=32,shuffle=False)
valid_data = valid_datagen.flow_from_directory('tmp/validation/', class_mode='categorical', target_size=(224,224), batch_size=32,shuffle=False)
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

print(resnet_base.summary())
model = models.Sequential()
model.add(resnet_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(4, activation = 'softmax'))
print(model.summary())

for layer in resnet_base.layers[:]:
  layer.trainable = False

print(model.summary())

learning_rate = 1e-4
model.compile(optimizer=optimizers.RMSprop(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

training_dir = '/tmp/train/'
testing_dir = '/tmp/test/'
validation_dir = '/tmp/validation/'

callbacks_list = [callbacks.ModelCheckpoint(
        filepath = 'resnet-finetune-model.h5',
        monitor = 'val_loss',
        save_best_only = True),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            mode='min',
            min_lr=1e-8),
        callbacks.CSVLogger(
            filename='resnet-finetune-model.csv',
            separator = ',',
            append = False)]

batch_size = 32
history = model.fit(train_data,
                             steps_per_epoch=4000//batch_size,
                             epochs=5,
                             validation_data=valid_data,
                             validation_steps=2217//batch_size,
                             callbacks=callbacks_list)

#resnet_base.trainable = True
for layer in resnet_base.layers[:165]:
  layer.trainable = False
for layer in resnet_base.layers[165:]:
  layer.trainable = True

for i, layer in enumerate(resnet_base.layers):
  print(i, layer.name, layer.trainable)

model.compile(optimizer=optimizers.RMSprop(lr=learning_rate),
              loss='categorical_crossentropy',
             metrics=['accuracy'])
print(model.summary())
print('Fit the model...')
t0 = time() #timing counter starts
print('The model has started learning...')
nepochs=50
batch_size=32
history=model.fit(train_data, #Learning process starts
                  steps_per_epoch=4000//batch_size,
                  epochs=nepochs,
                  validation_data=valid_data,
                  validation_steps=2217//batch_size,
                  callbacks=callbacks_list)
print('Fit model took', int(time() - t0),'s') #time is calculated with the help of counter

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.minorticks_on()
plt.grid(True)
plt.legend(['train', 'validation'], loc='upper left')
# save image to disk
plt.savefig('RESNET Final Model Accuracy', dpi=250)
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.minorticks_on()
plt.grid(True)
plt.legend(['train', 'validation'], loc='upper left')
# save image to disk
plt.savefig('RESNET Final Model Loss', dpi=250)
plt.show()

validation_datagen = ImageDataGenerator(rescale = 1./255)
evaluate_datagen = validation_datagen.flow_from_directory(validation_dir, class_mode='categorical', target_size=(224,224), batch_size=1,shuffle=False)

print('Validate the model')
final_result = model.evaluate(
    evaluate_datagen,
    steps = 2217)

t0 = time()
evaluate_datagen.reset()
val_predict = model.predict(
    evaluate_datagen,
    steps = 2217,
    verbose = 1)
print('Time taken to evaluate the model:',int(time()-t0),'seconds')

validation_samples = val_predict.shape[0]
print('Number of data points in validation set:',validation_samples)

val_predicted_classes = np.argmax(val_predict, axis = 1)
val_true_classes = evaluate_datagen.classes
val_class_labels = list(evaluate_datagen.class_indices.keys())

validation_report = metrics.classification_report(val_true_classes, val_predicted_classes, target_names=val_class_labels)
print('The validation report is as follows:')
print(validation_report)
val_conf_matrix = tf.math.confusion_matrix(labels=val_true_classes, predictions=val_predicted_classes).numpy()
print(val_conf_matrix)

figure1 = plt.figure()
val_conf_matrix = val_conf_matrix.astype('float') / val_conf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(val_conf_matrix, annot = True, cmap=plt.cm.Greens)
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted label')
plt.savefig('Validation data Confusion Matrix', dpi=250)
plt.show()

testing_datagen = ImageDataGenerator(rescale = 1./255)
test_data_datagen = testing_datagen.flow_from_directory(testing_dir, class_mode='categorical', target_size=(224,224), batch_size=1,shuffle=False)

print('Testing the model')
final_result = model.evaluate(
    test_data_datagen,
    steps = 2128)

test_data_datagen.reset()
predict_on_test_data = model.predict(
    test_data_datagen,
    steps = 2128,
    verbose = 1
)

print('Number of data points in test set:',predict_on_test_data.shape[0])

predicted_classes = np.argmax(predict_on_test_data,axis=1)
predict_true_classes = test_data_datagen.classes
predict_class_labels = list(test_data_datagen.class_indices.keys())

test_report = metrics.classification_report(predict_true_classes, predicted_classes, target_names = predict_class_labels)
print(test_report)

test_confusion_matrix = tf.math.confusion_matrix(labels=predict_true_classes,predictions = predicted_classes).numpy()
print(test_confusion_matrix)

figure1 = plt.figure()
test_confusion_matrix = test_confusion_matrix.astype('float') / test_confusion_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(test_confusion_matrix, annot = True, cmap=plt.cm.Greens)
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted label')
plt.savefig('Test data Confusion Matrix', dpi=250)
plt.show()


model_json = model.to_json()
with open("resnet50model.json","w") as json_file:
  json_file.write(model_json)
  #serializing the weights to HDF5
model.save('RESNET-Final-Model.h5')
print('Model saved to the disk')



son_file = open('resnet50model.json','r')
loaded_json_model = json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_json_model)
loaded_model.load_weights("RESNET-Final-Model.h5")
print("Loaded the model from disk")
