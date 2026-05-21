# AGENTS.md

## 项目约定

- 修改代码前先看现有实现，尽量沿用当前风格和接口组织。
- FA 相关代码目前约定 `B=1`，布局使用 `BHSD`，头维度固定为 `128`。
- 如果调整 FA 接口，测试代码、CPU 参考实现和 cuDNN 调用需要同步更新。
- cuDNN 的 shape、stride 计算优先使用 `int64_t`，避免乘法中间结果溢出。

## 格式约定

- 修改文本文件时保留该文件既有换行格式，不做无关的整文件行尾转换。
- 项目根构建文件（例如 `CMakeLists.txt`、`cmake_config.cmake`）当前使用 CRLF 行尾；修改后保持 CRLF，避免变成 CRLF/LF 混合行尾。
- `src/`、`test/` 下现有 C/C++/CUDA 源码和头文件当前主要使用 CRLF 行尾；新增同类文件也使用 CRLF。
- `AGENTS.md` 当前使用 LF 行尾；修改后保持 LF。
- `CuAssembler/` 下文件行尾历史上同时存在 LF 和 CRLF；修改时以目标文件当前格式为准，不跨目录批量统一。

## Git 约定

- 实现新功能前，先切换到 `main` 并执行 `git pull`，再从最新 `main` 切出新分支。
- commit message 使用中文。
- 只能推送 `codex/` 开头的远端分支。
- 不要直接推送 `main`。
- 分支名保持简短明确，例如 `codex/support-multi-head-fa`。
