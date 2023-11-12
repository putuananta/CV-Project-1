# Gender Classification 
Upload by: I Putu Ananta Yogiswara
Created by: CV B Oppenheimer, in 2023

```text
1. Hendra (VGG16, VGG19)
2. Fatturahman (VGG16, VGG19)
3. Dani (GoogleNet)
4. I Putu Ananta Yogiswara (GoogleNet)
5. Harrison (ResNet)
6. Fitrah (ResNet)
```
This GitHub repository focuses on gender classification using four different methods: VGG16, VGG19, ResNet, and GoogleNet. The project utilizes the CelebA dataset for training and testing. In this GitHub I am more emphasize using GoogleNet

## GoogleNet Details

For GoogleNet, the project emphasizes detailed preprocessing. The dataset is balanced using undersampling for equal representation of male and female data. The data is then split into training and testing sets. In the training dataset, data augmentation techniques such as RandomRotation and RandomHorizontalFlip are applied to enhance model generalization. Additionally, data normalization is performed for both training and testing datasets.

## Dataset
CelebA

## Preprocessing Steps for GoogleNet
```text
1. Undersampling for dataset balance
2. Splitting data into train and test sets
3. Training dataset transformations:
   a. RandomRotation
   b. RandomHorizontalFlip
   c. Normalization
4. Test dataset transformations:
   d. Normalization
```


## Model Architecture
```text
1. Utilizes pre-trained weights for GoogleNet
2. Fully connected layer structure:
   Three dense layers with ReLU activation 
   One dense layer with a sigmoid activation function
```

## Training Parameters
```text
1. Loss function: Binary Cross Entropy
2. Optimizer: Adam
3. Learning rate: 0.001
4. Epochs:
    15 epochs
    30 epochs

Both 15 and 30 epochs yield an impressive accuracy of 97%.
```

## Evaluation and Challenges
During dataset evaluation, mislabeling issues were identified. This emphasizes the importance of data quality and labeling accuracy in achieving reliable model performance.

## Dependencies

This lab requires **Python 3.7.3** and the following Python libraries installed:

* Basic Libraries: [NumPy](http://www.numpy.org/), [Matplotlib](http://matplotlib.org/)
* Domain-specific Libraries: [OpenCV](https://opencv.org/)
* Deep-learning Frameworks: [Keras](https://keras.io/), [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/)

📨 That's all, for any discussion kindly contact me here: ananta.putu@gmail.com
