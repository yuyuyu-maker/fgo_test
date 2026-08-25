# FPO Go2 C++ deploy notes
#
# Do NOT reinvent DDS/ONNX Runtime here. Use the maintained official stack:
#   https://github.com/unitreerobotics/unitree_rl_lab
#   (clone via: bash scripts/clone_unitree_rl_lab.sh)
#
# Mapping onto our FPO Go2 flat policy (see configs/fpo_go2_aligned.yaml):
#   obs:  float[48]
#   act:  float[12]  -> q_des = q_default + 0.25 * act
#   hz:   policy 50, lowcmd 500; kp=25, kd=0.5
#
# Modes (web UI badge):
#   sport — SportClient.Move (built-in gait)
#   fpo   — this C++ path; UDP JSON on port 18080
# Idle / no keys: {"vx":0,"vy":0,"yaw":0,"active":false}

参考目录（克隆后）:
  thirdparty/unitree_rl_lab/deploy/robots/go2/

构建示例（在 rl_lab 内，以官方 README 为准）:
  cd thirdparty/unitree_rl_lab/deploy/robots/go2
  mkdir -p build && cd build
  cmake .. && make
  ./go2_ctrl --network eth0
