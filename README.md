# Animal Image Classification


### Project Summary:

This project focuses on developing a **machine learning model** using **PyTorch** to classify animal images into **three distinct categories — Panda, Cat,** and **Dog.**
The model leverages **deep learning techniques** for feature extraction and image classification, providing accurate predictions across multiple image inputs.

---

### 📸 Dataset Description:

The dataset comprises a total of **2,400 labeled images** across **three animal classes — Panda, Cat,** and **Dog.**
Each image is preprocessed and standardized for consistent model training and evaluation.

### Dataset Split:
**Training Set:** 2,100 images located in the Train/ directory
**Validation Set:** 300 images located in the Valid/ directory

![](./visuals/cat-dog-panda-class-image.png?raw=true)

---

### Machine Learning Model Architecture:

The model is a custom **Convolutional Neural Network (CNN)** developed from scratch using PyTorch, designed to classify animal images into three categories: Panda, Cat, and Dog. It is built with two primary components — a **feature extraction block** and a **classification head**, each crafted to optimize performance and generalization.

The **feature extraction block** consists of four convolutional layers with progressively increasing filter depths of 64, 128, 256, and 512. Each convolutional layer is followed by **batch normalization** to stabilize and accelerate training, a **LeakyReLU** activation function (with a negative slope of 0.1) to prevent neuron inactivation, and a **max-pooling** layer to reduce spatial dimensions and capture dominant features. The final layer employs **adaptive average pooling** with a 1×1 output size, effectively compressing the spatial dimensions while preserving global feature representations.

Following the extraction block, the classification head transforms the learned features into categorical predictions. It begins by flattening the pooled feature maps into a one-dimensional vector, then passes through two **fully connected (linear)** layers. The first linear layer reduces the dimensionality from 512 to 256 neurons, activated by **ReLU**, while **dropout layers** (with rates of 0.5 and 0.3) are strategically placed to minimize overfitting. The final linear layer outputs three logits corresponding to the three target classes — Panda, Cat, and Dog. Overall, this architecture efficiently balances model complexity and computational efficiency, enabling accurate and robust animal image classification.

---

### Data Augmentation:

To enhance generalization and reduce overfitting, several **data augmentation techniques** were applied during training. These include **random horizontal flips, rotations, cropping, resizing**, and **normalization** to standardize image dimensions and pixel values. Such transformations increase dataset diversity and help the model become more robust to variations in orientation, lighting, and scale.

---

### Training Hyperparameters:

* Epochs: `120`
  
* Optimizer: `Adam`

* Initial Learning Rate: `0.001`
  
* Weight Decay: `1e-4`

* Batch size: `8`

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

The configurations discussed above, yielded a score of **0.965** on the Kaggle's Leaderboard.

![](./visuals/animal_classification_kaggle_leaderboard.png?raw=true)
