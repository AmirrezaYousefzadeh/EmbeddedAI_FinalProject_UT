# Step 1: train the baseline model

This folder contains a notebook that trains a simple CNN for CIFAR-10 image classification with grayscale input.
Upload it to the UT Jupyter Lab (choose a system with a GPU) and follow the notebook.

Train with TensorFlow 2.11 (Python 3.7 to 3.10). The boards and the Vitis AI 2.5 tools need a Keras 2 `.h5` model;
models saved by newer TensorFlow versions (Keras 3, for example on Google Colab) cannot be loaded there.

The result is `cifar10_bw_model.h5`, the input for step 2 (CPU inference on the Kria) and step 3a (quantization on
the CAES server).
