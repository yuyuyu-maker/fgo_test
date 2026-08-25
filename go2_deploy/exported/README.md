# Exported ONNX policies (deploy artifacts)

Put `policy.onnx` here so **runtime only needs the `go2_deploy/` tree**
(plus `.venv` / unitree SDK). Training `.pt` under `isaaclab_experiments/logs/`
is **not** required on the robot.

Layout:

```
exported/<model_name>/policy.onnx
```

Examples:

```
exported/baseline/policy.onnx
exported/reflow/policy.onnx
```

Export (on the training machine):

```bash
bash scripts/export_fpo_onnx.sh --model baseline
```

Then copy / rsync the whole `go2_deploy/` folder to the deploy host and run:

```bash
python -m go2_deploy --mode fpo --model baseline
```
