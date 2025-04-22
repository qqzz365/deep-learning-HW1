# Image Classification with PyTorch
This repository contains two Jupyter Notebook files (taskA.ipynb and taskB.ipynb) that implement image classification tasks using PyTorch on the MiniImageNet dataset. The projects explore different convolutional neural network (CNN) architectures, channel configurations, and ablation studies to evaluate model performance.

# Project Structure
- taskA.ipynb: Implements a SimpleCNN and a DynamicCNN for image classification with varying input channel configurations (RGB, RG, R). The DynamicCNN uses a dynamic convolution layer to handle different input channels flexibly.
- taskB.ipynb: Implements a ResNet34 baseline and a simplified ResNet (SimpleResNet) model, along with ablation studies to evaluate the impact of architectural components such as residual connections, batch normalization, max pooling, and additional residual blocks.
- requirements.txt: Lists the Python dependencies required to run the notebooks.
dataset/: Directory where the MiniImageNet dataset should be placed (not included in the repository; see Dataset Setup for details).

# Requirements
To run the notebooks, ensure you have Python 3.8 or later installed. Install the required dependencies using:
pip install -r requirements.txt


# Dataset Setup
Both notebooks assume the MiniImageNet dataset is used, which should be placed in the dataset/ directory. The dataset should include:
- images.zip: A zip file containing the image files.
- train.txt, val.txt, test.txt: Text files listing image paths and their corresponding labels.
For taskA.ipynb, additional text files (train_rgb.txt, train_rg.txt, train_r.txt, val_rgb.txt, val_rg.txt, val_r.txt) are generated or required for dynamic channel experiments.

To set up the dataset:
Place images.zip in the project root directory.
Run the notebooks, which will automatically extract images.zip to the dataset/ directory.
Ensure the text files are correctly formatted with image paths and labels (e.g., image_path label per line).

Note: The MiniImageNet dataset is not included in this repository. You can obtain it from MiniImageNet or other reliable sources.
Running the Notebooks


# taskA.ipynb
Objective: Train and evaluate SimpleCNN and DynamicCNN models on MiniImageNet with different input channel configurations (RGB, RG, R).
Key Features:
SimpleCNN: A basic CNN with two convolutional layers, trained separately for each channel configuration.
DynamicCNN: Uses a dynamic convolution layer to adapt to varying input channels (RGB, RG, R) within a single model.
Includes data preprocessing, model training, testing, and evaluation with metrics like accuracy and classification reports.
Computes model complexity (FLOPs and parameters) using ptflops and torchsummary.
Output: Test accuracies for each channel configuration, model summaries, and saved model weights.

# taskB.ipynb
Objective: Compare a ResNet34 baseline with a simplified ResNet (SimpleResNet) and conduct ablation studies to analyze architectural components.
Key Features:
ResNet34Baseline: A standard ResNet34 model without pretrained weights, trained on MiniImageNet.
SimpleResNet: A lightweight ResNet with fewer residual blocks for efficiency.
Ablation Studies: Evaluates the impact of removing residual connections, max pooling, batch normalization, and adding an extra residual block.
Computes model complexity (FLOPs and parameters) using ptflops and torchsummary.
Output: Training/validation/test accuracies, model summaries, and ablation study results comparing performance and complexity.
