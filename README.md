# Brain CT Image Hemorrhage Classification & Segmentation

**Date:** April 20, 2025
**Authors:** Pratosh Karthikeyan and Nirmal

## Abstract

This project constructed classification and segmentation models for brain CT images based on CNN (Convolutional Neural Network) and U-Net architecture. Six classes are involved based on different categories of hemorrhage in patient's brain, whereas the segmentation aims to provide the exact location of the hemorrhage for a given brain CT image.

### Methodology
- Cropped 3D data of brain CT scan to fit ventricular region compactly
- Used 3D U-net as semantic segmentation with weighted binary crossentropy loss

### Results
- **Classification Accuracy:** 68% (testing set)
- **Segmentation Accuracy:** 92% (testing set)

## Clinical Importance

Brain surgery requires doctors to precisely locate the treatment part, since a tiny error may cause serious results. Mathematical models can make better judgments for treatment.

### Tasks Addressed
1. Decide if there is hemorrhage in patient's brain
2. Determine which types of hemorrhage exist  
3. Locate exact position of hemorrhage

## Dataset

### Source
**Provider:** Zeta Surgical

**Company Mission:** Democratize access to accurate, safe, and fast image guidance, to unlock the use of image guidance directly at the point of care, and to enable new treatments in cases such as emergencies and bedside procedures.

### Specifications
- **Total Instances:** 752,803
- **Image Size:** 512 × 512 pixels
- **Channels:** 3 (RGB)

### Categories
| Category | Instances |
|----------|-----------|
| Epidural | 1,694 |
| Intraparenchymal | 15,664 |
| Intraventricular | 9,878 |
| Subarachnoid | 16,423 |
| Subdural | 32,200 |
| Multiple Hemorrhage | 32,074 |

### Preprocessing Steps
1. **Convert RGB to Grayscale:** Using formula `Xij = (Rij + Gij + Bij) / 3`
2. **Resize Images:** From 512×512 to 128×128 pixels
3. **Dataset Selection:** Random selection of 1,000 images per category for training

## Models

## Classification Model

### Architecture: CNN
- **Layers:** 
  - 3 convolution layers with ReLU and Max Pooling
  - 1 convolution layer with flattening layer
  - 2 fully-connected hidden layers
- **Parameters:** 1.2 million

### Dataset Split
- **Training Instances:** 4,800 (80%)
- **Testing Instances:** 1,200 (20%)

### Challenges
- Significant overfitting (90% training vs 50% validation accuracy)
- Unstable results due to random initialization dependency

### Solutions Attempted
- Added Dropout layers
- Increased dataset to 1,600 images per type
- Tuned learning rate and architecture size
- Changed dropout parameters

### Results
- **Final Testing Accuracy:** 68%
- **Final Training Accuracy:** 75%
- **Optimal Epochs:** 25
- **Maximum Validation Accuracy:** 70%

## Segmentation Model

### Architecture: 3D U-Net

### Dataset
- **Source:** Reference [4] - 3D MRI images
- **Total Images:** 87
- **Original Dimensions:** (176, 208, 176)
- **Cropped Dimensions:** (80, 120, 120)
- **Focus Area:** Ventricular region

### Architecture Details
- **Input Size:** 512 × 512
- **Structure:**
  - Contraction path (gray) - usual CNN process
  - Expansion path (black) - upsampling back to original size
- **Layers:**
  - Conv3D layers with 3×3×3 filters
  - 3D max-pooling with 2×2×2 cube
  - Concatenation of contracting and expanding paths

### Training Configuration
- **Parameters:** 1.2 million
- **Optimizer:** Adam
- **Learning Rate:** 2e-4
- **Loss Function:** Weighted binary crossentropy
- **Loss Weights:** [0.2, 0.8] (background, mask)
- **Metrics:** Accuracy

### Results
- **Testing Accuracy:** 92%
- **Training Accuracy:** 98%

## Performance Analysis

### Classification
**Challenges:**
- Much more difficult task compared to segmentation
- Required extensive trial and error
- Low and unstable accuracy results

### Segmentation
**Advantages:**
- More stable training process
- Good accuracy achieved

**Limitations:**
- Not good F1 score due to high class imbalance

## Future Work

### Proposed Improvements
- Use different weights for weighted binary crossentropy loss
- Analyze effect on confusion matrix
- Perform data augmentation to increase dataset size
- Fine-tune hyperparameters
- Use bigger architecture (current: 1.4M parameters)

## Acknowledgments

We hereby especially thank **Professor He Wang** for providing the opportunity for us to work on this project and to visit the company. We are also grateful for **Zeta Surgical**, for giving us such an opportunity to work on live industrial project by generously offering us its first-hand dataset.

**Special Thanks:**
- **Raahil Sha** (Co-Founder and CTO at Zeta Surgical) and his team
- Zeta Surgical team for providing first-hand dataset
- Company visit opportunity and access to high-power computers

## References

1. Intracerebral Haemorrhage Segmentation in Non-Contrast CT
2. Automatic Segmentation of Intracerebral Hemorrhage from Brain CT Images  
3. Segmentation and quantification of intra-ventricular/cerebral hemorrhage in CT scans
4. Segmentation part from MATH7243-2020 course
5. Performance evaluation of 2D and 3D segmentation approaches on CT images
6. MRI-Image-Segmentation
