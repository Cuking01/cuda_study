# AGENTS.md

## 项目约定

- 修改代码前先看现有实现，尽量沿用当前风格和接口组织。
- FA 相关代码目前约定 `B=1`，布局使用 `BHSD`，头维度固定为 `128`。
- 如果调整 FA 接口，测试代码、CPU 参考实现和 cuDNN 调用需要同步更新。
- cuDNN 的 shape、stride 计算优先使用 `int64_t`，避免乘法中间结果溢出。

## Git 约定

- 只能推送 `codex/` 开头的远端分支。
- 不要直接推送 `main`。
- 分支名保持简短明确，例如 `codex/support-multi-head-fa`。
