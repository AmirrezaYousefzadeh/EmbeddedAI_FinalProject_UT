# Step 3b (optional): your own DPU bitstream (CAES server)

The Kria has an FPGA. To run a network on it you load an overlay with a DPU (Deep Learning Processing Unit). The
course boards already have the default KV260 DPU (`dpu.bit` of pynq-dpu 2.5). In this step you build your own DPU
configuration and bitstream.

Ask yourself which DPU configuration suits your model best: are most of your layers supported by the DPU (see
`inspect_results.txt` from step 3a), and which data and operations does it expect? Options are in `dpu_conf.vh`
(see [PG338](https://docs.amd.com/r/en-US/pg338-dpu)).

## Get DPU-PYNQ 2.5 (matches PYNQ 3.0.1 and pynq-dpu 2.5 on the boards)

```bash
git clone --branch v2.5.1 https://github.com/Xilinx/DPU-PYNQ.git
cd DPU-PYNQ/boards
git clone https://github.com/Xilinx/XilinxBoardStore -b 2022.1
```

## Set up Vivado/Vitis 2022 and XRT on the server

```bash
module load xilinx/vivado/2022.2
module load xilinx/vitis/2022.2
module load xilinx/xrt
source /opt/xilinx/xrt/setup.sh        # only if XILINX_XRT is still empty after the module load
```

DPU-PYNQ 2.5 checks for Vitis **2022.1** in `check_env.sh`. With 2022.2 it stops with
"Please source Vitis 2022.1 settings". Change the version test in `boards/check_env.sh` from `2022.1` to `2022.2`.

## Configure and build

Edit `kv260_som/dpu_conf.vh` (and `kv260_som/prj_config` if needed), then build. This takes a while.

```bash
make BOARD=kv260_som
```

The result is in `kv260_som/`: `dpu.bit`, `dpu.hwh` and `dpu.xclbin`. Copy all three files **without renaming
them** into a subfolder of your Kria project folder, for example `my_dpu/`, and set `BITSTREAM = "my_dpu/dpu.bit"`
in the notebook (pynq-dpu 2.5 only uses an `.xclbin` file named `dpu.xclbin`). Compile your model for this DPU
with its `arch.json` (step 3a).

## Optional: inspect the design in Vivado

```bash
module load xilinx/vivado/2022.2
vivado kv260_som/binary_container_1/link/vivado/vpl/prj/prj.xpr
```
