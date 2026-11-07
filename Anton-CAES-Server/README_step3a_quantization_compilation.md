# Step 3a: quantization and compilation (CAES server)

This folder contains `vitis-ai-quantization_grayscale.py`, which does post-training quantization of the `.h5`
model with Vitis AI 2.5.

## Steps

### Copy the trained model and the script to the server

```bash
ssh your_username@anton.ewi.utwente.nl mkdir -p project
scp cifar10_bw_model.h5 vitis-ai-quantization_grayscale.py your_username@anton.ewi.utwente.nl:~/project/
# a whole folder: scp -r my_folder your_username@anton.ewi.utwente.nl:~/project/
```

### Get the dataset (CIFAR-10, python version)

```bash
cd ~/project
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz          # creates cifar-10-batches-py/
```

The download server can be slow; if you already have the file, copy it with `scp` instead.

### Start Vitis AI 2.5 with TensorFlow 2

```bash
apptainer run /remote/labware/containers/vitis-ai_2.5.sif
conda activate vitis-ai-tensorflow2
```

### Post-training quantization

```bash
python vitis-ai-quantization_grayscale.py --model cifar10_bw_model.h5 --data cifar-10-batches-py
```

This writes `inspect_results.txt` (which layers the DPU supports) and `cifar10_bw_model_quant.h5`, and prints the
accuracy of the quantized model on the test set. Options: `--calib N` (number of calibration images, default
1000), `--out DIR`, `--dump`, and `--target` (DPU name or `arch.json` path for the inspector, default
`DPUCZDX8G_ISA1_B4096`, the default KV260 DPU).

### Compile for the default KV260 DPU (`dpu.bit` of pynq-dpu)

```bash
vai_c_tensorflow2 -m cifar10_bw_model_quant.h5 \
    -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
    -o compiled -n cifar10_bw_model_compiled
```

Copy the result `compiled/cifar10_bw_model_compiled.xmodel` to the Kria, next to `kria_inference_DPU.ipynb`.

### Compile for your own DPU (step 3b)

Use the `arch.json` of your own DPU build (search for it in the build folder:
`find DPU-PYNQ/boards/kv260_som -name arch.json`). Rerun the script with `--target <path/to/your/arch.json>` so
that `inspect_results.txt` describes your DPU, then compile into a separate folder:

```bash
vai_c_tensorflow2 -m cifar10_bw_model_quant.h5 -a <path/to/your/arch.json> -o compiled_my_dpu -n cifar10_bw_model_compiled
```

On the Kria, put `compiled_my_dpu/cifar10_bw_model_compiled.xmodel` in the folder of your DPU files (`my_dpu/`,
see step 3b) and set `XMODEL = "my_dpu/cifar10_bw_model_compiled.xmodel"` in the notebook.
