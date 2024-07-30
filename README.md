# Optimizing JPEG Image Compression Using Reinforcement Learning: A Proximal Policy Optimization Approach

This repository is a proof of concept for optimizing JPEG image compression using reinforcement learning. The proposed method uses the Proximal Policy Optimization (PPO) algorithm to optimize the JPEG parameters used in compression. The goal is to find the optimal set of JPEG parameters that minimizes the distortion in the compressed image while maintaining a certain level of compression ratio.

## Custom Model training

`custom_model.ipynb` contains the code for training the custom model using the PPO algorithm. The custom model is a convolutional neural network that takes the image features and a compression ratio as input and outputs the optimal set of JPEG parameters. The model is trained using the PPO algorithm to minimize the distortion in the compressed image while maintaining a certain level of compression ratio.

## Evaluation

`benchmarks.ipynb` contains the code for evaluating the compression algorithms on a test dataset.
