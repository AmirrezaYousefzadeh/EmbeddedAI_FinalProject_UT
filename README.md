# Embedded AI: final project baseline (University of Twente)
### *Running an image classification model on the edge with hardware–software co-optimization*

Template code and setup for the final project of the Embedded AI course at the University of Twente. Use it as a
**baseline**, then optimize, experiment and improve the model with what you learn in the course.

---

## Project flow and where to find what

| Step | Where | Folder | Main files |
|---|---|---|---|
| 1. Train the network (CIFAR-10, grayscale) | UT Jupyter Lab (TensorFlow 2.11) | [Utwente-Jupyter-Notebook](Utwente-Jupyter-Notebook) | `Image_classification_model.ipynb` → `cifar10_bw_model.h5` |
| 2. CPU inference (TensorFlow, batch size 1) | Kria KV260 | [Kria](Kria) | `kria_inference_CPU.py` |
| 3a. Quantize and compile (Vitis AI 2.5) | CAES server | [Anton-CAES-Server](Anton-CAES-Server) | `vitis-ai-quantization_grayscale.py`, `vai_c_tensorflow2` → `.xmodel` |
| 3b. Own DPU bitstream (optional) | CAES server | [Anton-CAES-Server](Anton-CAES-Server) | DPU-PYNQ 2.5 → `dpu.bit/.hwh/.xclbin` |
| 3c. DPU inference | Kria KV260 | [Kria](Kria) | `kria_inference_DPU.ipynb` |

Each folder has its own `README.md` with the commands.

Tool versions (keep them for compatibility): TensorFlow 2.11 (Keras 2 `.h5`), Vitis AI 2.5, Vivado/Vitis 2022.x,
PYNQ 3.0.1 with pynq-dpu 2.5 on the boards.

All steps use the **same preprocessing**: the `rgb_to_gray` function (integer arithmetic, identical results
everywhere), then division by 255. If you change it, change it in every step. Otherwise the steps report different
accuracies for reasons unrelated to the model.

---

## Course flexibility

The services and tools shown here are **suggestions**, not rules. For example, you may train on another platform,
as long as the model is saved with TensorFlow 2.11 (a Keras 2 `.h5` file). Models saved by newer TensorFlow
versions, such as the one on Google Colab, cannot be loaded by the boards or by the Vitis AI 2.5 tools.

> This repository is a **vanilla baseline**. You are encouraged to implement your own hardware–software
> co-optimizations with methods from the course or from external documentation.

---

## Video playlist (setup guide)

[![Embedded AI setup playlist](https://img.youtube.com/vi/6HrpetCuJoA/hqdefault.jpg)](https://www.youtube.com/playlist?list=PLo2ZwbR4rgUSKURdz-35U5dcxfc9D-buz)

[Watch the playlist on YouTube](https://www.youtube.com/playlist?list=PLo2ZwbR4rgUSKURdz-35U5dcxfc9D-buz)

The videos show an earlier version of the setup. Differences on the current boards: TensorFlow is preinstalled
and the proxy is already configured for pip/apt/git (see [Kria/README.md](Kria/README.md)), and the scripts take
the dataset path as an argument. Use the code in this repository, not the repository linked in the video
descriptions.

---

## Further reading and optimization references (optional)

| Topic | Resource |
|-------|-----------|
| Neural architecture search benchmark | [NATS-Bench](https://github.com/D-X-Y/NATS-Bench) |
| The DPU | [DPUCZDX8G product guide (PG338)](https://docs.amd.com/r/en-US/pg338-dpu) |
| Vitis AI 2.5 | [User guide (UG1414)](https://docs.amd.com/r/2.5-English/ug1414-vitis-ai/Vitis-AI-Overview) and [GitHub repository](https://github.com/Xilinx/Vitis-AI/tree/v2.5) |
| Vitis AI tutorial | [XUP Vitis AI tutorial](https://xilinx.github.io/xup-vitis-ai-tutorial/index.html) |
| Optimization ideas | Quantization-aware training, pruning, transfer learning, neural architecture search |

---

Questions about the repository: contact the teaching team.

Acknowledgement: the first version of this material was written by Pratik Phadte.
