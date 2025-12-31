import os
import numpy as np
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#surpress tf gpu warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


EPOCHS = 10 #number of times model sees entire dataset
#dimensions to which image will be scaled
IMG_WIDTH = 128 
IMG_HEIGHT = 128
#images processed per training step
BATCH_SIZE = 32

#paths
TRAIN_DIR = "../chest_xray/chest_xray/train"
TEST_DIR = "../chest_xray/chest_xray/test"



def get_model():
    #load base CNN pretrained on ImageNet without top classificaiton layer
    base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3) #3 channels for rgb
    )

    # Freeze base model first (avoid overfitting)
    base.trainable = False


    #add classification layer on top of base
    model = models.Sequential([
        base,           #pretrained feature extractor from densenet121
        layers.GlobalAveragePooling2D(), #reduce spatial dimensions
        layers.Dense(256, activation="relu"),  #fully connected layer
        layers.Dropout(0.5),        #reduce overfitting
        layers.Dense(1, activation="sigmoid") #binary classification
    ])

    #compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model



def main():

    # Augment data to make model more robust and reduce overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        validation_split=0.2
    )

    #create generator for training images
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        #resize
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'       #use this for training
    )

    #create generator for validation images
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'     #use this for validation
    )


    
    #create generator for test images without augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False       #use this for testing, keep original order for evaluation
    )


    # Compute class weights (Pneumonia dataset is heavily imbalanced)
    classes = train_generator.classes
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes
    )
    #convert to dictionary format
    class_weights = dict(enumerate(class_weights))

    model = get_model()

    # stop training early if validation loss doesnt improve and reduce learning rate if plateua
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
    ]

    # Train
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print("\nTraining complete.\n")

    # Evaluate
    test_loss, test_acc, test_auc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
