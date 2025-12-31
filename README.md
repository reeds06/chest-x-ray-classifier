# chest-x-ray-classifier
Deep learning CNN using TensorFlow implementing data augmentation, class weighting, and staged fine-tuning to classify chest X-rays as pneumonia vs. healthy.

**Overview:**

This project is a deep learning pipeline that classifies chest X-ray images as pneuomonia or healthy. I used DenseNet121, a CNN used for image recognition, with some light data augmentation to minimize overfitting and class weighting to deal with the imbalanced dataset. 


**Dataset:**

Kaggle Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Contains almost 6,000 organized into train, test, and val directories labeled as normal or pneuomonia

**Pipeline:**

**1. Data Loading:
**  - Chest X-Ray images are loaded from directories labeled normal and pneuomonia

**2. Data Preprocessing:
**  - Images are resized to 128x128 to be passed into CNN
  - Pixel values are normalized to [0,1]

**3. Data Augmentation (Training Only):
**  - Random rotations, zooms, and shifts are applied to input images to make model more robust and reduce overfitting.

**4. Model Training:
**  - DenseNet121, a CNN used for image recognition, was trained on the augmented training images
    - Class weights are computed from training data and applied during training to account for imbalanced dataset
    - Certain optimizations are included, such as early stop

**5. Evaluation:**
    - The model is then tested on testing data, and its accuracy is printed

