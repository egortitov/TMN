import tensorflow as tf
from keras import layers, models
from keras.applications import VGG19, ResNet50
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# GPU configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# Loading data
train_dir = './train'
val_dir = './val'
test_dir = './test'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(120, 120),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(120, 120),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(120, 120),
    batch_size=batch_size,
    class_mode='binary'
)

# Fully connected network
model_dense = models.Sequential()
model_dense.add(layers.Flatten(input_shape=(120, 120, 3)))
model_dense.add(layers.Dense(256, activation='relu'))
model_dense.add(layers.Dense(128, activation='relu'))
model_dense.add(layers.Dense(64, activation='relu'))
model_dense.add(layers.Dense(1, activation='sigmoid'))

# Convolutional neural network
model_conv = models.Sequential()
model_conv.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 3)))
model_conv.add(layers.MaxPooling2D((2, 2)))
model_conv.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_conv.add(layers.MaxPooling2D((2, 2)))
model_conv.add(layers.Flatten())
model_conv.add(layers.Dense(64, activation='relu'))
model_conv.add(layers.Dense(1, activation='sigmoid'))

# Transfer learning for VGG19
base_model_vgg = VGG19(weights='imagenet', include_top=False, input_shape=(120, 120, 3))
base_model_vgg.trainable = False

model_transfer_vgg = models.Sequential()
model_transfer_vgg.add(base_model_vgg)
model_transfer_vgg.add(layers.Flatten())
model_transfer_vgg.add(layers.Dense(256, activation='relu'))
model_transfer_vgg.add(layers.Dense(1, activation='sigmoid'))

# Transfer learning for ResNet
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(120, 120, 3))
base_model_resnet.trainable = False

model_transfer_resnet = models.Sequential()
model_transfer_resnet.add(base_model_resnet)
model_transfer_resnet.add(layers.Flatten())
model_transfer_resnet.add(layers.Dense(256, activation='relu'))
model_transfer_resnet.add(layers.Dense(1, activation='sigmoid'))

# Compilation and training of models
model_dense.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_dense = model_dense.fit(train_generator, epochs=2, validation_data=val_generator)

model_conv.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_conv = model_conv.fit(train_generator, epochs=2, validation_data=val_generator)

model_transfer_vgg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_transfer_vgg = model_transfer_vgg.fit(train_generator, epochs=2, validation_data=val_generator)

model_transfer_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_transfer_resnet = model_transfer_resnet.fit(train_generator, epochs=2, validation_data=val_generator)

# Increased number of epochs and plotting learning curves
def plot_learning_curves(history, title):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Increased epochs to 8 for demonstration
history_dense = model_dense.fit(train_generator, epochs=8, validation_data=val_generator)
plot_learning_curves(history_dense, 'Dense Model Learning Curves')

history_conv = model_conv.fit(train_generator, epochs=8, validation_data=val_generator)
plot_learning_curves(history_conv, 'Convolutional Model Learning Curves')

history_transfer_vgg = model_transfer_vgg.fit(train_generator, epochs=8, validation_data=val_generator)
plot_learning_curves(history_transfer_vgg, 'VGG19 Transfer Learning Curves')

history_transfer_resnet = model_transfer_resnet.fit(train_generator, epochs=8, validation_data=val_generator)
plot_learning_curves(history_transfer_resnet, 'ResNet Transfer Learning Curves')

# Remember to close the session when done
sess.close()