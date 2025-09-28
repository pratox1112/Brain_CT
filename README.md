# 🧠 Brain CT Hemorrhage Classification and Segmentation

This repository contains the code and methodology for a multi-task deep learning model that performs **simultaneous classification and segmentation** of brain hemorrhages from CT images. The model was trained on clinical data provided by Zeta Surgical and leverages convolutional neural networks (CNNs) to identify various hemorrhage types and segment lesion areas.

## 🚀 Project Overview

- **Framework:** PyTorch  
- **Tasks:** Multi-label classification and binary segmentation  
- **CT Views Used:** Brain Bone Window  
- **Model Type:** Shared encoder with classification and segmentation heads  
- **Dataset:** Annotated CT scans with classification labels and polygonal segmentation masks  

> 📌 Note: The `'multi'` class label was excluded from active learning in some stages of the pipeline.

---

## 📂 Dataset

- 6 hemorrhage classes: `epidural`, `subdural`, `intraparenchymal`, `subarachnoid`, `intraventricular`, `multi-type`
- Polygon-based segmentation masks available for a subset of the dataset
- A total of 5000 images were randomly sampled and split into 80% training and 20% validation sets

---

## 🧠 Model Architecture

The model consists of a shared convolutional encoder and two heads: one for classification and one for segmentation.

### Shared Encoder
- 3 convolutional layers with filter sizes 16, 32, and 64
- Each layer includes ReLU activation and max pooling
- Dropout (0.1) is applied after the first two convolutional layers for regularization

### Classification Head
- The encoder output is flattened and passed through a fully connected layer with 128 units
- Dropout (0.3) is applied before the final output layer
- The final layer contains 5 output units with sigmoid activation for multi-label classification

### Segmentation Head
- The encoder output is passed through three transposed convolution layers to upsample it
- A final sigmoid activation produces a pixel-wise binary segmentation mask

---

## 🏋️‍♂️ Training and Optimization

The model is trained using a multi-task learning approach with two separate loss objectives: one for classification and another for segmentation.

### Optimization Setup
- **Loss Function:** Binary Cross Entropy (BCE) for both tasks
- **Optimizer:** Adam
- **Learning Rate:** 0.0001
- **Epochs:** 20
- **Batch Size:** 16
- **Hardware:** Trained on CPU

### Training Process
- In each epoch, the model performs forward passes on training data and computes losses for both classification and segmentation outputs
- The total loss is the sum of classification and segmentation loss
- Gradients are computed and used to update model weights using backpropagation
- Validation is run after each epoch using the same loss metrics but without gradient updates
- Training and validation loss values are tracked and plotted to monitor convergence

---

## 📈 Results

- **Classification Accuracy:** 96.10%  
- **Segmentation IoU:** 1.0  
- The training and validation loss curves show rapid convergence and consistent generalization

---

## 🔬 Discussion

- The shared encoder allows for efficient learning of features common to both tasks
- Dropout layers helped improve generalization, especially given the limited dataset size
- High IoU in segmentation suggests strong spatial accuracy, though limited mask data may have contributed
- Using only a single CT view (brain bone window) proved sufficient for both classification and segmentation

---

## 🔮 Future Work

- Extend model to use multiple CT window types (e.g., brain, subdural, max contrast)
- Explore more advanced architectures like UNet or attention-based decoders
- Apply data augmentation to simulate real-world variability in CT imaging
- Train on GPU for deeper, more expressive models
- Validate performance in real clinical settings with expert annotations

---


## 🧾 Citation

**Pratosh Karthikeyan & Nirmalkumar Thirupallikrishnan Kesavan**  
*College of Engineering, Northeastern University, 2025*
