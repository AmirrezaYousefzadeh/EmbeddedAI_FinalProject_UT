# Kria KV260: running the model on the board

This folder covers the steps that run on the AMD Kria KV260 board:

- **Step 2, CPU inference:** the trained Keras model (`.h5`) runs with TensorFlow on the board's ARM cores
  (`kria_inference_CPU.py`).
- **Step 3c, DPU inference:** the quantized and compiled model (`.xmodel`) runs on the DPU, the deep-learning
  accelerator in the FPGA, through PYNQ (`kria_inference_DPU.ipynb`).

## 1. Your board

- Every group has its own board, `caes-kriaN.ewi.utwente.nl`, where **N** is your board number. Open
  `http://caes-kriaN.ewi.utwente.nl` in a browser (UT network or UT VPN) and log in with the password you received
  from the teaching team. Do not share the password outside your group.
- You get the PYNQ Jupyter environment and work as `root`. Your files are in `/root/jupyter_notebooks`, the
  Jupyter start folder. For a terminal, choose *New → Terminal*.
- **Keep your own copy of your work** (download it, or push it to a git repository). Before every course, the
  boards are reset to a clean installation, and everything stored on them is deleted.
- Everything the course needs is installed. Please do not change the system configuration (proxy, Jupyter, SSH).
- If your board does not respond, a monitoring service restarts it automatically within about 30 minutes. If it
  is still unreachable after that, or if the software on it no longer works, contact the teaching team. They can
  reset the board; your files on the board are then lost.

## 2. Internet access

The boards have no direct internet access. Web traffic goes through the UT proxy `proxy.utwente.nl:3128`.
`pip`, `apt` and `git` are already configured to use it. Other programs (`curl`, `wget`, Python `requests`) need
the proxy as an option or as environment variables in the terminal:

```bash
export http_proxy=http://proxy.utwente.nl:3128 https_proxy=http://proxy.utwente.nl:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy no_proxy=localhost,127.0.0.1
curl -sI https://pypi.org | head -n 1      # prints an "HTTP/... 200" line when the proxy works
```

## 3. Python environments

The board has two separate Python environments. Keep them separate: PYNQ needs its own versions of NumPy and
other packages.

| Environment | Use it for | Jupyter kernel | Terminal |
|---|---|---|---|
| PYNQ | the DPU (step 3c) | *Python 3 (ipykernel)* | active by default |
| TensorFlow 2.11 (CPU), in `/usr/local/share/tf-venv` | TensorFlow on the ARM cores (step 2) | *TensorFlow 2.11 (CPU)* | `source /usr/local/share/tf-venv/bin/activate` |

Check the TensorFlow environment:

```bash
source /usr/local/share/tf-venv/bin/activate
python3 -c "import tensorflow as tf, numpy as np; print(tf.__version__, np.__version__)"   # 2.11.0 1.26.x
deactivate
```

If `/usr/local/share/tf-venv` does not exist on your board, create the environment yourself (this takes about
10 minutes):

```bash
python3 -m venv /usr/local/share/tf-venv && source /usr/local/share/tf-venv/bin/activate
pip install "tensorflow==2.11.0" "numpy<2.0" ipykernel
python3 -m ipykernel install --prefix /usr/local --name tf-2.11 --display-name "TensorFlow 2.11 (CPU)"
deactivate
```

## 4. Dataset

Download the CIFAR-10 *python version* on the board and unpack it in your project folder:

```bash
mkdir -p ~/jupyter_notebooks/project && cd ~/jupyter_notebooks/project
curl -L -x http://proxy.utwente.nl:3128 -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz          # creates cifar-10-batches-py/
```

The download server can be slow (up to half an hour for the 163 MB). If you already have the file, upload it
through Jupyter instead and only run the `tar` command.

Put `kria_inference_CPU.py`, `kria_inference_DPU.ipynb` and your model files in the same folder (upload them in
Jupyter, or `git clone` this repository on the board).

## 5. Step 2: CPU inference with TensorFlow

```bash
cd ~/jupyter_notebooks/project
source /usr/local/share/tf-venv/bin/activate
python3 kria_inference_CPU.py --model cifar10_bw_model.h5 --data cifar-10-batches-py --num_images 100
python3 kria_inference_CPU.py --model cifar10_bw_model.h5 --num_images full    # final benchmark: all 10 000 images
deactivate
```

The script classifies the images one by one (batch size 1). It runs one warm-up inference first and then times
the loop over the images (model call and argmax). It prints the accuracy, the total time, the average time per image and the throughput.

## 6. Step 3c: DPU inference with PYNQ

Open `kria_inference_DPU.ipynb` with the PYNQ kernel (*Python 3 (ipykernel)*). Put next to it:

- your compiled model `cifar10_bw_model_compiled.xmodel` (made on the CAES server, see
  [../Anton-CAES-Server](../Anton-CAES-Server));
- only if you built your own DPU (step 3b): its `dpu.bit`, `dpu.hwh` and `dpu.xclbin`, not renamed, in a subfolder
  such as `my_dpu/`. Set `BITSTREAM = "my_dpu/dpu.bit"` in the notebook (pynq-dpu 2.5 only uses an `.xclbin`
  named `dpu.xclbin`). Do not put a file named `dpu.bit` directly next to the notebook: it would replace the
  default DPU.

With the default `BITSTREAM = "dpu.bit"`, pynq-dpu loads its standard DPU for the KV260 (DPUCZDX8G, B4096). The
`.xmodel` must be compiled for the DPU that you load.

Only one program at a time can use the FPGA. If loading the overlay fails, shut down the other notebooks that use
it (*Running* tab in Jupyter) and restart the kernel.

## 7. Preprocessing

All steps use the same preprocessing: `rgb_to_gray` (integer arithmetic, so the result is identical in every
environment), then division by 255. If you change it, change it in every step. A difference in preprocessing is a
common cause of unexplained accuracy differences between the steps.

## 8. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | Activate the TensorFlow environment, or select the *TensorFlow 2.11 (CPU)* kernel (section 3). |
| `ModuleNotFoundError: No module named 'pynq'` | Use the PYNQ kernel *Python 3 (ipykernel)*, not the TensorFlow one. |
| `curl` or `wget` hangs or cannot connect | Add the proxy (`curl -x http://proxy.utwente.nl:3128 ...`) or set the variables of section 2. |
| The DPU notebook fails at the overlay or model step | Check that the `.xmodel` was compiled for the loaded DPU, and that no other notebook uses the FPGA. |
| The board does not respond | Wait 30 minutes (automatic restart). Still down: contact the teaching team. |
