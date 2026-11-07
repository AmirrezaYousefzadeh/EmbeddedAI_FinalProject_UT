"""Step 2: CIFAR-10 inference on the Kria KV260 ARM CPU with TensorFlow (batch size 1).

USAGE (on the board, in a Jupyter terminal):
    source /usr/local/share/tf-venv/bin/activate          # TensorFlow 2.11 environment of the course boards
    python3 kria_inference_CPU.py --model cifar10_bw_model.h5 --data cifar-10-batches-py --num_images 100
    python3 kria_inference_CPU.py --model cifar10_bw_model.h5 --num_images full     # all 10 000 test images
    deactivate

The preprocessing (rgb_to_gray + division by 255) must be identical in every step of the project (training,
quantization, CPU and DPU inference). Copy the same function if you change it.
"""
import argparse
import os
import pickle
from time import time

import numpy as np


def rgb_to_gray(x):
    """uint8 RGB images (..., 32, 32, 3) -> uint8 grayscale (..., 32, 32, 1). Integer arithmetic, so the result is
    bit-identical in every environment (NumPy, TensorFlow, Vitis AI container, PYNQ)."""
    x = x.astype(np.uint32)
    gray = (x[..., 0] * 299 + x[..., 1] * 587 + x[..., 2] * 114 + 500) // 1000
    return gray.astype(np.uint8)[..., np.newaxis]


def load_cifar10_test(directory):
    """Test batch of the CIFAR-10 python version (cifar-10-python.tar.gz -> cifar-10-batches-py/)."""
    with open(os.path.join(directory, "test_batch"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3) uint8 RGB
    y = np.array(batch[b"labels"])
    return x, y


parser = argparse.ArgumentParser(description="CIFAR-10 inference of a Keras .h5 model on the CPU")
parser.add_argument("--model", required=True, help="path to the Keras .h5 model")
parser.add_argument("--data", default="cifar-10-batches-py",
                    help="folder with the CIFAR-10 python batches (default: ./cifar-10-batches-py)")
parser.add_argument("--num_images", default="100", help="number of test images, or 'full' for all 10 000")
args = parser.parse_args()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # less TensorFlow start-up noise
import tensorflow as tf  # noqa: E402  (imported after the environment variable)

x_test, y_test = load_cifar10_test(args.data)
x_test_norm = rgb_to_gray(x_test).astype("float32") / 255.0
print(f"Test samples: {x_test_norm.shape[0]}, input shape {x_test_norm.shape[1:]}")

model = tf.keras.models.load_model(args.model, compile=False)
print(f"Model '{args.model}' loaded.")

num_images = x_test_norm.shape[0] if args.num_images.lower() == "full" else min(int(args.num_images), x_test_norm.shape[0])

# One warm-up inference: the first call builds the TensorFlow graph and would distort the timing.
model(x_test_norm[:1], training=False)

print(f"Running inference on {num_images} images (batch size 1)...")
y_pred = np.empty(num_images, dtype=np.int64)
start = time()
for i in range(num_images):
    # model(...) instead of model.predict(...): predict() adds tens of ms of framework overhead per call,
    # which is not part of the network's latency.
    probs = model(x_test_norm[i:i + 1], training=False)
    y_pred[i] = int(np.argmax(probs))
stop = time()

correct = int(np.sum(y_pred == y_test[:num_images]))
total_time = stop - start
print(f"Accuracy: {100.0 * correct / num_images:.2f}% ({correct}/{num_images})")
print(f"Total execution time: {total_time:.4f} s")
print(f"Average inference time per image: {1000.0 * total_time / num_images:.2f} ms")
print(f"Throughput: {num_images / total_time:.2f} FPS")
