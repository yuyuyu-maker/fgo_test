# thirdparty

构建 `g1_ctrl` 需要 ONNX Runtime **1.22.0**（与 `robots/g1_29dof/CMakeLists.txt` 路径一致）：

```bash
bash scripts/download_onnxruntime.sh
```

目录结构：

```
thirdparty/onnxruntime-linux-x64-1.22.0/
  include/
  lib/libonnxruntime.so.1.22.0
```

C++ **unitree_sdk2** 默认安装到系统 `/usr/local`（见 `scripts/setup_unitree_sdk2.sh`），不放在本目录。
