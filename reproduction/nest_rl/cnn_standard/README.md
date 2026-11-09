# 标准环境 CNN 路线

此目录预留给使用标准 Overcooked 环境和空间观测编码的 CNN 实验。目前尚未实现训练代码或复现结果；后续任务见[复现总览](../README.md)。

在仓库根目录执行以下命令，为标准路线创建独立的 Python 3.10 环境。已有 `.conda/` 时直接复用。

```powershell
Set-Location reproduction/nest_rl/cnn_standard
conda create --prefix .conda python=3.10 pip -y
conda run --prefix .conda python --version
```
