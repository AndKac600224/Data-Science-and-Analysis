## Collaborators:
* @AndKac600224
* 
*
*
*

# Image Classification using Convolutional Neural Networks (CNN)

This project implements an image classification system using Convolutional Neural Networks (CNN) in **TensorFlow/Keras**.  
It is designed to classify images from a custom dataset and demonstrates a full pipeline: training, validation, testing, and evaluation.

## **Project Structure**
Brain_Cancer_TF_AI/   
├── train_model.py # Training script  
├── eval_model.py # Evaluation script  
├── dataset/ # User-provided dataset (not included in repo)  
│ ├── training/ # Training images organized by class  
│ ├── validation/ # Validation images organized by class  
│ └── testing/ # Test images organized by class  
└── README.md # this file  

> **Note:** The dataset is **not included** in this repository due to its size. To get the dataset, click on URL below and download data to your local device  

https://drive.google.com/drive/folders/19FRUtOCDf6yd7PUP5lGurRDn0h1uYFAO?usp=sharing

To upload data with Python use this commands:
```bash
import gdown

# ID_PLIKU to część linku pomiędzy /d/ a /view
url = 'https://drive.google.com/uc?id=ID_PLIKU'
output = 'dataset'
gdown.download(url, output, quiet=False)
```

### **Dataset Requirements**

The dataset must follow this structure:
dataset/  
├── training/  
│ ├── yes/  
│ ├── no/  
├── validation/    
│ ├── yes/  
│ ├── no/  
└── testing/  
├── yes/  
├── no/  

- Each subfolder should contain images of a single class.  
- The scripts automatically read all images in the provided paths.

## Train the model

Run the training script:
`train_model.py`

This script will:
* Load training and validation datasets from your local folders.
* Create and compile the CNN model.
* Train the model with early stopping and save the best model as best_model_postx.keras.
> **Note** This may take some time to compile (finish the training), so be sure that your CPU is able to deal with operations like that.

## Evaluate the model

After training, run the evaluation script:
`eval_model.py`

This script will:
* Load the trained model from best_model_postx.keras.
* Evaluate the model on training, validation, and testing sets.
* Display a confusion matrix and a detailed classification report.

## Model Architecture

The CNN model consists of:
* 3 Convolutional layers with ReLU activations.
* MaxPooling layers after each convolution.
* Flatten layer followed by Dense layers with Dropout to prevent overfitting.
* Output layer with Softmax activation (number of neurons = number of classes).

## Metrics tracked during training:
* Accuracy
* Precision
* Recall

## Usage Notes
* Ensure that your dataset paths match the expected structure in train_model.py and eval_model.py.
* You can experiment with hyperparameters such as learning rate, batch size, and epochs inside the training script.
* For large datasets, training may take significant time; GPU is recommended.
* 

## References
* https://www.tensorflow.org/guide/keras?hl=pl
* https://www.tensorflow.org/tutorials/images/cnn?hl=pl
* https://keras.io/api/data_loading/image

## Example outputs
<img width="564" height="407" alt="image" src="https://github.com/user-attachments/assets/3d04d44f-58f3-4612-b251-458143c29e05" />
<img width="777" height="622" alt="image" src="https://github.com/user-attachments/assets/5cb23451-47ad-424d-aa36-805950cb74c3" />

## Summary   
Authors: Kacper Andrzejewski and collaborators mentioned before
Project Type: Deep Learning / Image Classification / AI / Convolutional Neural Network

