# cuda_study_dls

## 简介

这个项目用于学习和优化 CUDA 算子，当前重点是针对 RTX 5060 Ti / 消费级 Blackwell 架构做 SGEMM、HGEMM、FlashAttention 以及一些底层指令和访存微基准实验。代码以手写 CUDA kernel 为主，配套 CPU 参考实现、cuBLAS / cuDNN 对照实现和测试程序，用于验证正确性、统计耗时并对比性能表现。

当前 CMake 默认编译架构在 `cmake_config.cmake` 中配置为 `86 120f`。如果你的 CUDA / CMake 版本不支持 `120f`，请参考本文后面的 CMake 配置问题。

## 目录结构

```text
.
├── CMakeLists.txt          # 主构建文件，定义 cu_operator、cu_expriment 和各测试可执行文件
├── cmake_config.cmake      # 本机编译器路径和 CUDA 架构配置
├── src/                    # 算子实现和公共工具
│   ├── sgemm.cu/.h         # FP32 GEMM 多版本实现，以及 cuBLAS 对照入口
│   ├── hgemm.cu/.h         # FP16 GEMM 多版本实现，以及 cuBLAS 对照入口
│   ├── fa.h                # FlashAttention 对外接口
│   ├── fa/                 # FlashAttention v1-v4、自定义 barrier/TMA helper、cuDNN frontend 对照实现
│   ├── expr/               # cp.async、ldg128、lsu、wavefront 等微基准实现
│   └── tool/               # CUDA 内存、stream、event、同步和错误检查封装
├── test/                   # 测试和 benchmark 入口
├── analyzer/               # SASS / cubin 分析相关实验材料
├── ref/                    # CUDA、Nsight Compute、架构白皮书等参考资料
├── CuAssembler/            # 本仓库内的 CUDA SASS 汇编/反汇编工具代码
└── cudnn-frontend/         # NVIDIA cudnn-frontend submodule
```

## 主要文件

- `src/sgemm.cu` / `src/sgemm.h`：SGEMM kernel 的多个优化版本，测试入口会和 CPU 参考实现、部分 cuBLAS 入口做对照。
- `src/hgemm.cu` / `src/hgemm.h`：HGEMM kernel 的多个优化版本，使用 `half` 数据类型。
- `src/fa/fa_v1.cu` 到 `src/fa/fa_v4.cu`：FlashAttention 的多个手写实现版本。当前 FA 代码约定 batch size 为 1，输入布局为 `BHSD`，head dim 固定为 128。
- `src/fa/fa_cudnn.cu`：基于 `cudnn_frontend` 的 SDPA 对照实现，使用 causal mask 和 FP32 中间计算。
- `test/fa.cu`：FA 正确性和性能测试入口，会测试 `fa_v1` 到 `fa_v4` 以及 `fa_cudnn`。
- `test/sgemm.cu`、`test/hgemm.cu`：GEMM 正确性和性能测试入口。
- `test/*.cu`、`test/*.cpp`：指令吞吐、MMA latency、PCIe bandwidth、warp shuffle、cp.async 等实验入口。

## 外部依赖

需要本机已安装：

- NVIDIA GPU 和驱动。
- CUDA Toolkit，需包含 `nvcc`、CUDA runtime、CUDA driver API、cuBLAS、NVRTC 等库。
- 支持 `120f` 架构的 CUDA / CMake 组合，或者按下文临时绕过 `120f` 配置问题。
- CMake。
- GCC / G++。默认路径在 `cmake_config.cmake` 中写为 `/usr/bin/gcc` 和 `/usr/bin/g++`。
- cuDNN，包含 `cudnn.h` 和 `libcudnn`。如果不在默认路径，请设置 `CUDNN_ROOT`。
- Python 3、`sympy`、`pyelftools`。只有使用 `CuAssembler` 相关脚本时才需要。

`CMakeLists.txt` 会查找 CUDA Toolkit 和 cuDNN，并把 `cudnn-frontend/include` 加入 include path。cuBLAS 来自 CUDA Toolkit，用于 GEMM 对照测试。

## Submodule

本项目使用一个 git submodule：

```text
cudnn-frontend -> https://github.com/NVIDIA/cudnn-frontend
```

首次 clone 后执行：

```bash
git submodule update --init --recursive
```

如果你是用 `--recursive` clone 的仓库，则通常不需要再执行一次：

```bash
git clone --recursive <repo-url>
```

## 构建

先确认 `cmake_config.cmake` 中的编译器路径和 CUDA 架构适合本机：

```cmake
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CUDA_ARCHITECTURES 86 120f)
```

然后配置和编译：

```bash
cmake -S . -B build
cmake --build build
```

也可以只编译某个测试目标：

```bash
cmake --build build --target test_fa
cmake --build build --target test_sgemm
cmake --build build --target test_hgemm
```

## 运行

构建完成后，测试程序位于 `build/` 下。常用入口：

```bash
./build/test_fa
./build/test_sgemm
./build/test_hgemm
```

实验类入口包括：

```bash
./build/test_lsu
./build/test_wavefront
./build/test_cp_async
./build/test_ldg128
./build/test_pcie_bandwidth
./build/test_ex2_fp16x2_throughoutput
./build/test_ex2_fp32_throughoutput
./build/test_shfl
./build/test_mma_m16n8k16_latency
./build/test_mma_m16n8k16_latency_ipl2
```

如果机器上有多张 GPU，可以用 `CUDA_VISIBLE_DEVICES` 指定设备：

```bash
CUDA_VISIBLE_DEVICES=0 ./build/test_fa
```

`test_fa` 默认会先跑较小规模的正确性测试，再对 `n = 2^14` 到 `2^17`、`heads = 6/36` 的数据集测试 `fa_v3`、`fa_v4` 和 `fa_cudnn` 的速度。FA 当前默认 head dim 为 128，batch size 为 1。

## CMake 配置问题

当前遇到过一个 CMake 配置问题：配置阶段可能报不支持 `120f`。

临时绕过方法：

1. 先把 `cmake_config.cmake` 里的

   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 86 120f)
   ```

   改成：

   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 86 120)
   ```

2. 执行一次配置：

   ```bash
   cmake -S . -B build
   ```

3. 再把 `120` 改回 `120f`，重新配置：

   ```bash
   cmake -S . -B build
   ```

这样通常可以让后续构建正常识别 `120f`。
