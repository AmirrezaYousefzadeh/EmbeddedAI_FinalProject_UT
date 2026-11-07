"""Step 3a: post-training quantization of the grayscale CIFAR-10 Keras model with Vitis AI 2.5 (TensorFlow 2).

Run inside the Vitis AI 2.5 container, conda environment vitis-ai-tensorflow2 (see README_step3a_quantization_compilation.md):
    python vitis-ai-quantization_grayscale.py --model cifar10_bw_model.h5 --data cifar-10-batches-py

Outputs (in --out, default: the current folder):
    inspect_results.txt   which layers the DPU supports (Vitis inspector)
    <name>_quant.h5       quantized model, input for the compiler vai_c_tensorflow2
    dump_results/         (optional) layer dumps for debugging

The preprocessing (rgb_to_gray + division by 255) must be identical in every step of the project.
"""
import argparse
import os
import pickle

import numpy as np
import tensorflow as tf


def rgb_to_gray(x):
    """uint8 RGB images (..., 32, 32, 3) -> uint8 grayscale (..., 32, 32, 1), integer arithmetic (same in all steps)."""
    x = x.astype(np.uint32)
    gray = (x[..., 0] * 299 + x[..., 1] * 587 + x[..., 2] * 114 + 500) // 1000
    return gray.astype(np.uint8)[..., np.newaxis]


def load_cifar10(directory):
    def load_batch(name):
        with open(os.path.join(directory, name), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        return batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(batch[b"labels"])
    parts = [load_batch(f"data_batch_{i}") for i in range(1, 6)]
    x_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    x_test, y_test = load_batch("test_batch")
    return (x_train, y_train), (x_test, y_test)


parser = argparse.ArgumentParser(description="Vitis AI post-training quantization (grayscale CIFAR-10 model)")
parser.add_argument("--model", required=True, help="trained Keras model (.h5)")
parser.add_argument("--data", default="cifar-10-batches-py", help="folder with the CIFAR-10 python batches")
parser.add_argument("--out", default=".", help="output folder")
parser.add_argument("--calib", type=int, default=1000, help="number of training images used for calibration")
parser.add_argument("--target", default="DPUCZDX8G_ISA1_B4096",
                    help="DPU target for the inspector (default: the standard KV260 DPU, B4096)")
parser.add_argument("--dump", action="store_true", help="also dump layer outputs (debugging)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

tf_model = tf.keras.models.load_model(args.model, compile=False)
tf_model.summary()

(x_train, y_train), (x_test, y_test) = load_cifar10(args.data)
x_train_norm = rgb_to_gray(x_train).astype("float32") / 255.0
x_test_norm = rgb_to_gray(x_test).astype("float32") / 255.0
y_test_onehot = tf.one_hot(y_test.astype(np.int32), depth=10)
print("train:", x_train_norm.shape, "test:", x_test_norm.shape)

from tensorflow_model_optimization.quantization.keras import vitis_inspect, vitis_quantize  # noqa: E402

inspector = vitis_inspect.VitisInspector(target=args.target)
inspector.inspect_model(tf_model,
                        input_shape=x_train_norm.shape[1:],
                        dump_model=True,
                        dump_model_file=os.path.join(args.out, "inspect_model.h5"),
                        dump_results=True,
                        dump_results_file=os.path.join(args.out, "inspect_results.txt"),
                        verbose=0)

quantizer = vitis_quantize.VitisQuantizer(tf_model)
tfq_model = quantizer.quantize_model(calib_dataset=x_train_norm[:args.calib])
name = os.path.splitext(os.path.basename(args.model))[0]
tfq_path = os.path.join(args.out, f"{name}_quant.h5")
tfq_model.save(tfq_path)
print("quantized model saved to", tfq_path)

# Accuracy of the quantized model on the full test set (should be close to the float model)
tfq_model = tf.keras.models.load_model(tfq_path)
tfq_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])
tfq_model.evaluate(x_test_norm, y_test_onehot, batch_size=1000)

if args.dump:
    quantizer.dump_model(tfq_model, dataset=x_train_norm[0:1], output_dir=os.path.join(args.out, "dump_results"),
                         dump_float=True)
