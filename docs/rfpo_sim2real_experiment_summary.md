# 跟学长同步一下 RFPO / Reflow 这边的理解和实验进展

------

## 一、我目前的理解（导向对不对想跟你确认）

我理解的 RFPO 相对纯 FPO，重点不是仿真里 64 step 跑满能不能多拿一点分，而是 真机部署 / sim2real 这条线上能不能用。

- 训练侧：我们在 FPO 上加 Reflow 辅助损失，把 flow 路径拉直，这样 推理时可以少用 sampling steps。
- 但开 Reflow 之后，仿真训练回报普遍会比 baseline 低 1～2 分（Go2 2500 iter 大概 baseline 41.08，reward_aware 39.48，adaptive 39.93，theory 39.60）。我感觉这像是 Reflow 和主 CFM/FPO 目标之间的冲突，可以分析一下瓶颈，不一定说明 idea 本身没用。
- 所以真正要看的是 部署侧：少步推理（1/4/8 step）会不会塌、random 噪声模式下稳不稳、延迟能不能接受。真机算力有限，不能假设永远 64 step 满精度跑。

**sim2real 怎么衡量**，我还没完全想清楚，目前想法是：

1. **主指标应该是 step sweep**，看不同 sampling steps 下 return 怎么掉，尤其 random 模式（我觉得更接近真机有噪的情况）。这是我后面设计实验的主要依据。
2. 真机侧可能还要做对比：比如 **不微调直接部署** vs **采一点真机数据微调**，但采多少条、什么任务、怎么定义成功，我这边还没定。
3. 仿真里也可以加扰动（观测噪声、摩擦、延迟）当代理，但真机数据如果后面能采，应该更直接。

需要沟通的一个问题是，究竟要跑什么任务，我目前跑的有go2和spot，机器狗不在智平方这里，然后spot我还没有问李宴军学长，机械臂是可以确定的无法部署，那就是需要重新考虑一个仿真实验and采数据的方法

## 二、已经跑完的实验和结果

### 评测怎么做的（step sweep）

- `sampling_steps`：64 / 32 / 16 / 8 / 4 / 1
- `eval_modes`：zero（初始噪声 0）、random（随机噪声）
- `num_envs=2048`，每 env 1 episode

### Go2 结果

任务：`Isaac-Velocity-Flat-Unitree-Go2-v0`，2500 iter，`model_2499.pt`

**训练结束 mean reward（仿真在线 RL）**

| 变体 | mean reward |
|------|-------------|
| baseline | 41.08 |
| reward_aware | 39.48 |
| adaptive_compute | 39.93 |
| theory | 39.60 |

**Step sweep — mean episode reward（n=2048）**

| 变体 | mode | 64 | 32 | 16 | 8 | 4 | 1 |
|------|------|-----|-----|-----|-----|-----|-----|
| baseline | zero | 41.89 | 41.83 | 41.84 | 41.87 | 41.83 | **32.60** |
| baseline | random | 41.34 | 41.38 | 41.41 | 41.50 | 41.71 | **29.16** |
| reward_aware | zero | 41.78 | 41.74 | 41.77 | 41.73 | 41.76 | 41.73 |
| reward_aware | random | 39.56 | 39.64 | 39.65 | 39.66 | 39.60 | 39.69 |
| adaptive_compute | zero | 41.87 | 41.87 | 41.87 | 41.88 | 41.88 | 41.86 |
| adaptive_compute | random | 39.66 | 39.67 | 39.69 | 39.73 | 39.81 | 39.74 |
| theory | zero | 41.78 | 41.76 | 41.75 | 41.75 | 41.76 | 41.75 |
| theory | random | 39.75 | 39.73 | 39.71 | 39.75 | 39.79 | 39.68 |

**DROP（相对 64 step，同一变体同一 mode）**

| 变体 | mode | 32 | 16 | 8 | 4 | 1 |
|------|------|-----|-----|-----|-----|------|
| baseline | zero | -0.06 | -0.05 | -0.02 | -0.06 | **-9.29** |
| baseline | random | +0.04 | +0.07 | +0.16 | +0.38 | **-12.18** |
| reward_aware | zero | -0.03 | -0.00 | -0.05 | -0.02 | -0.04 |
| reward_aware | random | +0.08 | +0.10 | +0.11 | +0.04 | +0.13 |
| adaptive_compute | zero | -0.00 | -0.01 | +0.00 | +0.01 | -0.01 |
| adaptive_compute | random | +0.00 | +0.02 | +0.07 | +0.15 | +0.08 |
| theory | zero | -0.02 | -0.03 | -0.03 | -0.02 | -0.03 |
| theory | random | -0.02 | -0.04 | -0.00 | +0.04 | -0.07 |

一眼能看出来的：**baseline 在 random + 1 step 塌了**（41.34 → 29.16）；ideas 在 random 下 1～64 step 基本平。64 step 满步时 baseline random 仍最高。

### Spot 结果

任务：`Isaac-Velocity-Flat-Spot-v0`，1500 iter，`model_1499.pt`

**Step sweep — mean episode reward（n=2048）**

| 变体 | mode | 64 | 32 | 16 | 8 | 4 | 1 |
|------|------|------|------|------|------|------|------|
| baseline | zero | 347.85 | 348.86 | 347.06 | 349.24 | 348.04 | **62.26** |
| baseline | random | 334.72 | 336.58 | 336.46 | 337.55 | 339.40 | **25.31** |
| reward_aware | zero | 346.57 | 346.96 | 347.49 | 346.30 | 345.93 | 346.01 |
| reward_aware | random | 314.75 | 314.59 | 316.99 | 314.36 | 316.04 | 315.36 |
| adaptive_compute | zero | 352.12 | 351.65 | 349.97 | 349.99 | 347.68 | 353.46 |
| adaptive_compute | random | 317.96 | 322.31 | 321.98 | 320.67 | 321.78 | 321.57 |

**DROP（相对 64 step）**

| 变体 | mode | 32 | 16 | 8 | 4 | 1 |
|------|------|------|------|------|------|--------|
| baseline | zero | +1.00 | -0.80 | +1.39 | +0.19 | **-285.60** |
| baseline | random | +1.86 | +1.74 | +2.83 | +4.68 | **-309.40** |
| reward_aware | zero | +0.39 | +0.92 | -0.27 | -0.64 | -0.56 |
| reward_aware | random | -0.16 | +2.24 | -0.39 | +1.29 | +0.61 |
| adaptive_compute | zero | -0.47 | -2.14 | -2.13 | -4.44 | +1.34 |
| adaptive_compute | random | +4.36 | +4.02 | +2.71 | +3.82 | +3.62 |

Spot 上更明显：**满步 baseline random 最高（334.72），但 1 step 直接到 25.31**；reward_aware / adaptive_compute 在 1～64 step 全程大概 314～322，很平。

### 还在跑的

Spot **theory** 训练当时大概 1363/1500，训完 watcher 会自动在 GPU1 跑 theory 的 step sweep（`wait_spot_theory_then_step_sweep.sh`）

Go2 / Spot 其他变体的 step sweep 已经跑完了

### 训练时遇到的小问题

训练里自带的 post-training eval 经常在 headless 容器里 Vulkan 崩，所以改成训完单独跑 `eval_sampling_steps.py`

`fpo_operator` 训练塌了（nan），停了

`adaptive_compute` 训练里 `adaptive_mean_steps` 经常塌到 ~1.0（step predictor 的问题），但 Spot 上 eval 反而挺稳，这块我还没想通

## 三、我打算继续做的实验（验证设计）

我设计的验证实验分 **定量** 和 **定性** 两块，下面先列实验内容和指标，再按优先级排期。排期里的每一项都对应到上面的编号。

### 定量实验（机器狗 Go2 / Spot 为主）

| 编号 | 实验 | 做什么 | 主要看什么指标 | 状态 |
|------|------|--------|----------------|------|
| **A1** | **少步鲁棒性（step sweep）** | 同一 ckpt，扫 steps 64→1，zero + random，n=2048 | mean return \(R(k)\)；**DROP** \(\Delta(k)=R(k)-R(64)\)；**Retention** \(R(k)/R(64)\)；zero–random gap；collapse（\(R(1)/R(64)<0.7\) 或 episode length 断崖） | Go2 / Spot 大部分变体 **已跑完**；Spot theory **待训完** |
| **A2** | **推理延迟** | 固定 batch，测 `act_inference` 墙钟时间 | Latency(k) ms/forward；Speedup = Lat(64)/Lat(k)；能否 < 控制周期（如 50Hz → 20ms） | **未做** |
| **A3** | **路径是否更直（机制）** | theory 日志里的 path_straightness、discretization_gap；或离线算 \(\|a_{64}-a_k\|\) | discretization gap；path straightness；action jerk / Δa | 训练日志有部分；**系统 eval 未做** |
| **A4** | **仿真扰动（sim2real 代理）** | 观测噪声、质量/摩擦 ±10–20%、动作延迟 1–2 step | return retention；跌倒/早停率；episode length | **未做** |
| **A5** | **训练侧消融** | plain reflow vs ideas；`reflow_loss_coef` 扫 {0, 0.1, 0.3, 1.0} | 最终 \(R(64)\)、DROP 曲线、train time、adaptive_mean_steps | plain reflow **未完整跑**；coef **未做** |
| **真机（若有狗）** | 零样本部署 / 少步 vs 满步 | 固定速度指令跟踪、连续行走 | 速度 RMSE、行走时长、跌倒/干预次数、实测 Hz | **资源待定**（智平方无狗；Spot 要问李宴军学长） |

**A1 成功判据（我理解的）：** ideas 的 DROP 曲线比 baseline **更平**；random 模式下 1/4/8 step 仍可用；最好绝对分也不要明显输给 baseline。

### 定性实验

| 编号 | 做什么 | 看什么 | 状态 |
|------|--------|--------|------|
| **B1** | rollout 视频（同指令、同初始） | 少步是否抖、滑、跪地 | **未做** |
| **B2** | flow 轨迹可视化（`play_plot` 类） | 路径是否更直、少步是否偏离满步 | **未做** |
| **B3** | step–return 曲线图 | DROP 是否更缓（**主图**） | 有数字，**图还没画** |
| **B4** | Pareto 图（return vs latency） | 少步是否在质量–延迟上占优 | 依赖 A2 |

定性不替代 A1，主要是解释和汇报用。

### 机械臂（若后面要做 sim2real）

Realman 跟 Panda 不同构，不能硬套 FPO 仓 Can ckpt。定量看仿真/真机 success rate、Sim2Real gap、EE 误差、推理延迟；定性看成功/失败案例视频和失败模式分类。这块要等任务定义和资源。

---

### 按优先级排的（对应上面编号）

**P0（马上要做）**

- Spot theory 训完 → **A1** step sweep，看跟 reward_aware / adaptive 比怎么样（也接近 plain uniform reflow）

**P1（对照要补全）**

- **A5**：跑 plain Reflow（`reflow` 变体）同协议训练 + **A1** step sweep——区分「开 Reflow 的税」和「具体 idea 的增益」；Go2 目前没有 `unitree_go2_flat_flow_reflow` 完整 run
- **A2**：测 Latency(k)，配合 **B4** Pareto 图

**P2（如果时间够）**

- **A5**：`reflow_loss_coef` 消融（训练回报 vs DROP 怎么权衡）
- **A4**：仿真扰动 eval
- **A3**：gap / straightness 系统整理
- **B1–B3**：视频 + 轨迹 + 主曲线图

**P3（需要资源和任务定义）**

- 真机 Go2 / Spot：**A1 真机版**（零样本部署 + 跟踪/稳定性）
- 真机机械臂：少量 demo 微调对比（条数、任务待定）

### 主结果表打算怎么呈现（狗）

对每个变体、每个 mode 一行：

```
steps:     64    32    16     8     4     1
return:   ...
DROP:      0   -x    -y    ...
latency:  ...（A2 补）
```

再加：**random@1 相对 baseline@1 的 Δ**，以及 Pareto（return@k vs latency@k）。

------



## 四、任务类型和数据采集（这块想请你定）

我目前分三块理解：

### 1. 机器狗运动（Go2 / Spot）

- 仓库里是纯仿真在线 RL，**没有 BC、没有 demo 采集**
- 训完拿 `model_*.pt`，部署要 SDK + 观测对齐 + sim2real
- **智平凡这边我目前不知道有没有 Go2 / Spot 真机**——如果有，后面 sim2real 评测怎么做，得你那边确认



### 2. 机械臂 — FPO 仓库里已有的（Panda）

`manipulation_experiments` 里几个任务都能仿真跑通：

- Can、Square、BoxCleanup、LiftTray、Threading
- 机器人是 Panda（单臂或双臂），数据在 HF（robomimic / DexMG）
- 流程：LeRobot 数据 → `pretrain_flow_bc.py` BC → `finetune_online_rl.py` 仿真 FPO++ 微调
- **跟睿尔曼不同构**，ckpt 不能直接部署到智平凡真机



### 3. 机械臂 — 智平凡 / 睿尔曼

- 采集管线在 `realman_teleop_v2`：双臂 VR 遥操作，LeRobot 格式
- 典型 schema：28 维 action/state（每臂 14），相机 `left_wrist_rgb` / `right_wrist_rgb` / `high_rgb`，30 Hz
- RL infra 那边有杯 / 块 / 碗等多物体任务；BC 这边 **建议按任务切数据集**，混一起训容易糊（当前 BC 不用 language task 条件）
- 同构仿真可以用 Gazebo + `ros2_rm_robot` 采跟真机同 schema 的数据，场景要自己搭

**我想问学长的几件事：**

1. 智平凡机械臂 **具体做哪些任务**（杯/块/碗全做还是选几个）、成功怎么定义、单双臂怎么配——**这部分任务是我来定还是你来定？** 我这边可以按 schema 去采和执行，但任务清单和实验室要交什么，我觉得得你拍板。
2. 智平凡 **有没有 Go2 / Spot 真机** 可以做运动策略 sim2real？
3. 机械臂主线是 **只真机 BC**，还是 **Gazebo 同构仿真 + 真机** 一起做？

------



## 五、需要你帮忙确认的问题（汇总）

1. **导向**：RFPO 叙事以 step sweep（尤其 random @ 1/4/8 step）为主指标，sim2real 用真机评测 + 可选少量真机微调对比，这个方向 OK 吗？
2. **真机微调实验**：如果要采真机数据做微调对比，大概多少条 episode、什么任务算合理？还是先做零样本部署就够了？
3. **机械臂任务**：智平凡采什么、做什么任务，谁定？
4. **真机资源**：狗有没有、臂怎么排期？

------

有任务定义或优先级上的调整你直接在这份文档里批注，我按你的方向改实验排期。