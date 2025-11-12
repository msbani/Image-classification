# Animal Image Classification


### Project Summary:

This project builds a machine learning model to classify multiple animal images into 3 categories: Panda, Cat, Dog, using Pytorch.

---

### Dataset Description:

The dataset consists of `2100` images for **3 animal classes**. Here are some sample images for these three classes:

![](./visuals/cat-dog-panda-class-image.png?raw=true)

---

### Machine Learning Model Architecture:

This model is composed of a Neural Network from Scratch utilizing CNN architecture. 
First, added 5 `Feature Extraction Layers (Conv2d -> BatchNorm -> ReLU -> MaxPooling)`.
Then, used `Global Average Pooling Layer` and `Flattened the Tensor`.
Lastly, passed through the `Fully Connected Layers`.

---

### Data Augmentation:

For the training dataset, I applied the following data augmentation to avoid overfitting:

```
# Data Augmentation
transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
transforms.RandomRotation(degrees=15),  # Rotate images by up to 15 degrees
transforms.RandomResizedCrop(size=img_size[0], scale=(0.8, 1.0)),  # Randomly crop and resize images
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust colors
transforms.RandomGrayscale(p=0.1),      # Occasionally convert to grayscale
```

---

### Training Hyperparameters:

* Epochs: `100`
  
* Optimizer: `Adam`

* Initial Learning Rate: `1e-4`
  
* Weight Decay: `1e-4`

* Batch size: `32`

---

### Loss and Accuracy:

![](./visuals/animal_classification_loss_accuracy.png?raw=true)

---

### Inferences:

![](./visuals/animal_classification_inference-transformed.png?raw=true)

---

### Confusion Matrix:

![](./visuals/animal_classification_confusion_matrix.png?raw=true)

---

### Accuracy on Test Dataset for Kaggle Submission

The configurations discussed above, yielded a score of **0.89666** on the Kaggle's Leaderboard.

![](./visuals/animal_classification_kaggle_leaderboard.png?raw=true)
