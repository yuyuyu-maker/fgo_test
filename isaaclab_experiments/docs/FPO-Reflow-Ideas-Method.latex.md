# Method：面向流策略的在线 Reflow 扩展

我们在 FPO++ 之上引入一组**在线** Rectified Flow（Reflow）目标，用于 locomotion。所有变体共用同一套流策略 Actor–Critic 与 FPO trust-region 更新；仅通过辅助损失（或诊断量）把速度场拉向更直的传输，从而在推理时用更少欧拉步。

**符号。** 观测 \(o\)，动作 \(a\in\mathbb{R}^{d_a}\)，速度场 \(v_\theta(o,t,x)\)。时间参数化与本仓库 FPO 一致（与部分 RF 文献方向相反）：\(t=1\) 为噪声端，\(t=0\) 为动作端。默认欧拉预算 \(N=64\)。

---

## 1. 预备：条件流策略

### 1.1 路径与目标速度

给定噪声 \(x_0\sim\mathcal{N}(0,I)\) 与动作端点 \(x_1\)，直线路径为
\[
x_t = t\,x_0 + (1-t)\,x_1,\qquad
u^\star(x_0,x_1)=x_0-x_1.
\]
Actor 预测 \(v_\theta(o,t,x_t)\)。推理时从 \(t=1\) 积到 \(t=0\)，共 \(N\) 步欧拉：
\[
a = \mathrm{ODE}_{v_\theta}^{(N)}(o,x_0).
\]
记 \(T_v(o,x_0)\) 为场 \(v\) 下的 \(N\) 步端点映射。

### 1.2 基线 FPO++ 更新

每次迭代：

1. 用 \(\pi_\theta\) 采集 rollout（默认从 \(x_0\sim\mathcal{N}(0,I)\) 起步积分；部分变体改起点）。
2. 用 GAE 估计 advantage \(A\)。
3. 在 mini-batch 上最小化 FPO surrogate（基于 CFM 的策略比值 / ASPO trust region）与 value loss：
\[
\mathcal{L}_{\mathrm{FPO}}
= \mathcal{L}_{\mathrm{surr}} + \lambda_V\,\mathcal{L}_V.
\]

下文所有方法都是在同一网络上对 \(\mathcal{L}_{\mathrm{FPO}}\) 的**叠加项**。消融 `baseline` 仅用 \(\mathcal{L}_{\mathrm{FPO}}\)。

---

## 2. 在线 Reflow（均匀）

**动机。** 独立采样的 \((x_0,x_1)\) 易诱导弯曲传输，少步欧拉误差大。2-Rectified Flow [Liu et al., 2022] 先用当前场积分得到耦合端点，再在直线段上重训。我们把它改成**在线 RL 辅助项**，而非单独离线重训。

**过程（对观测 mini-batch \(\{o_i\}\)）。**

**算法 1 — 在线 Reflow**
1. 对每个 \(o_i\)，采样 \(n\) 个噪声 \(x_0^{i,k}\sim\mathcal{N}(0,I)\)。
2. **端点（截断梯度）：**  
   \(x_1^{i,k}\leftarrow T_{\bar v}(o_i,x_0^{i,k})\)，  
   其中 \(\bar v=v_\theta\)（online）或 EMA 副本（第 4 节）。
3. 按与 CFM 相同的 Beta 反 CDF 调度采样 \(t^{i,k}\)。
4. 构造 \(x_t^{i,k}=t^{i,k}x_0^{i,k}+(1-t^{i,k})x_1^{i,k}\)，目标 \(u^{i,k}=x_0^{i,k}-x_1^{i,k}\)。
5. 对**online** 场做回归：
\[
\ell_{i,k}
=\bigl\|v_\theta(o_i,t^{i,k},x_t^{i,k})-u^{i,k}\bigr\|^2_{\mathrm{red}},
\qquad
\mathcal{L}_{\mathrm{reflow}}
=\frac{1}{Bn}\sum_{i,k}\ell_{i,k},
\]
其中 \(\|\cdot\|_{\mathrm{red}}\) 为动作维上的 CFM 归约（默认对均方取平方根）。

**总损失。**
\[
\mathcal{L}
=\mathcal{L}_{\mathrm{FPO}}+\lambda_r\,\mathcal{L}_{\mathrm{reflow}}.
\]
默认 \(\lambda_r=1\)，\(n=4\)。均匀平均对应模式 \(\mathsf{uniform}\)。

*分工：* \(\mathcal{L}_{\mathrm{FPO}}\) 提升回报；\(\mathcal{L}_{\mathrm{reflow}}\) 改善路径几何（少步保真）。

---

## 3. Advantage 加权 Reflow

**动机。** 并非每个状态都值得同等直线化。高 \(A\) 区域应对齐「好」的传输；对低 \(A\) 强行掰直可能浪费容量。

**权重。** 对样本 \(i\)、advantage \(A_i\)、阈值 \(\tau\geq 0\)，
\[
w_i=\max(A_i-\tau,0),\qquad
\tilde w_i=\frac{w_i}{\sum_j w_j+\varepsilon}.
\]
若 \(\sum_j w_j\approx 0\)，退回均匀权重。

**损失。**
\[
\mathcal{L}_{\mathrm{reflow}}
=\sum_{i,k}\tilde w_i\,\ell_{i,k}.
\]
端点仍由 online actor 生成（不用 EMA）。对应模式 \(\mathsf{reward\_aware}\)。

---

## 4. 带 EMA 端点的算子式 Reflow

**动机。** 将 reflow 视为几何版**策略改进算子**：稳定的目标策略提出端点，online 策略拟合由其诱导的直线路径。

**端点 Actor。** 维护 actor 的 EMA 参数 \(\theta_{\mathrm{EMA}}\)。Warmup 结束后，取冻结副本 \(\bar v=v_{\theta_{\mathrm{EMA}}}\)，令
\[
x_1^{i,k}=T_{\bar v}(o_i,x_0^{i,k})\quad\text{（无梯度）}.
\]
速度回归**仅**用 \(v_\theta\)（online）；梯度不进入 \(\theta_{\mathrm{EMA}}\)。Warmup 内 \(\bar v\leftarrow v_\theta\)。

**正 Advantage 权重。**
\[
w_i=A_i\cdot\mathbf{1}[A_i>0],
\]
再按第 3 节归一化（全零则退回均匀）。对应模式 \(\mathsf{fpo\_operator}\)。

**正确性约束。** 端点前向必须走独立的 detached EMA 模块。禁止把 EMA 权重复制到训练 actor 上再算 reflow 后 restore：否则损失在 EMA 权重下计算、梯度却作用到 online 参数，计算图错位。

---

## 5. 自适应积分预算

**动机。** Reflow 把场掰直；另设一头在算力受限时决定**何时**可用更少步数。

**预测器。** 小型 MLP \(\phi_\psi(o)\) 在离散预算 \(\mathcal{K}=\{1,4,8,16,64\}\) 上输出 logits。令 \(\hat k(o)=\arg\max\phi_\psi(o)\)，并定义期望归一化预算（延迟代理）
\[
L_{\mathrm{lat}}(o)
=\mathbb{E}_{k\sim\mathrm{softmax}(\phi_\psi(o))}\!\Bigl[\frac{k}{\max\mathcal{K}}\Bigr].
\]

**零起点保真。** 与确定性零噪声评测对齐：
\[
a_N=\mathrm{ODE}_{v_\theta}^{(N)}(o,0)\quad\text{（截断梯度）},\qquad
a_{\hat k}=\mathrm{ODE}_{v_\theta}^{(\hat k)}(o,0),
\]
\[
\mathcal{L}_{\mathrm{fid}}=\bigl\|a_{\hat k}-a_N\bigr\|^2,\qquad
\mathcal{L}_{\mathrm{lat}}=\mathbb{E}\bigl[L_{\mathrm{lat}}(o)\bigr].
\]
可选 advantage 奖励项 \(-\beta\,\mathbb{E}[\mathrm{relu}(A)]\)（\(\beta=0.1\)）：
\[
\mathcal{L}_{\mathrm{ada}}
=\mathcal{L}_{\mathrm{fid}}+\alpha\,\mathcal{L}_{\mathrm{lat}}-\beta\,\mathbb{E}[\mathrm{relu}(A)].
\]
总损失加 \(\lambda_a\,\mathcal{L}_{\mathrm{ada}}\)（默认 \(\lambda_a=0.1\)，\(\alpha=1\)）。该预设同时打开均匀 reflow（第 2 节）。

---

## 6. Random-\(x_0\) 一致性

**动机。** 评测含两种起点：\(x_0=0\) 与 \(x_0\sim\mathcal{N}(0,I)\)。Reflow 改善零起点少步质量，但可能伤害随机起点满步回报。第 5 节自适应一致性仅用零起点，无法直接补随机缺口。

**采集混合。** 采集时以概率 \(p\) 用 random \(x_0\)，否则用 zero（\(p=0.5\)），使 buffer 覆盖两种评测协议。

**一致性损失。** 共享同一次随机抽样 \(x_0\sim\mathcal{N}(0,I)\)：
\[
a_N=\mathrm{ODE}_{v_\theta}^{(N)}(o,x_0)\quad\text{（截断梯度）},
\]
\[
\mathcal{L}_{\mathrm{rx0}}
=\frac{1}{|K|}\sum_{k\in K}
\bigl\|\mathrm{ODE}_{v_\theta}^{(k)}(o,x_0)-a_N\bigr\|^2,
\]
默认 \(K=\{1,4,8\}\)。总损失加 \(\lambda_x\,\mathcal{L}_{\mathrm{rx0}}\)（默认 \(\lambda_x=0.1\)）。无 step predictor；目标是随机起点下少步/满步一致，而非学习离散预算。

---

## 7. 总目标

各开关用指示函数写出：
\[
\begin{aligned}
\mathcal{L}
&=
\mathcal{L}_{\mathrm{FPO}}
+\mathbf{1}_r\,\lambda_r\,\mathcal{L}_{\mathrm{reflow}}
+\mathbf{1}_a\,\lambda_a\,\mathcal{L}_{\mathrm{ada}}
+\mathbf{1}_x\,\lambda_x\,\mathcal{L}_{\mathrm{rx0}}.
\end{aligned}
\]
\(\mathcal{L}_{\mathrm{reflow}}\) 内部由单一模式选择均匀 / advantage / 算子加权，并由标志选择 online 或 EMA 端点。

**算法 2 — 带扩展的一次 FPO 更新**
1. 采集 rollout；用 GAE 计算 \(A\)。
2. 对每个 mini-batch：
   1. 计算 \(\mathcal{L}_{\mathrm{FPO}}\)。
   2. 若开 reflow：按所选模式/端点 actor 跑算法 1，加 \(\lambda_r\,\mathcal{L}_{\mathrm{reflow}}\)。
   3. 若开自适应：加 \(\lambda_a\,\mathcal{L}_{\mathrm{ada}}\)。
   4. 若开 random-\(x_0\)：加 \(\lambda_x\,\mathcal{L}_{\mathrm{rx0}}\)。
   5. 对 online 参数（及若存在的 \(\psi\)）反传 \(\mathcal{L}\)。
3. 可选记录诊断量（第 8 节）；不参与梯度。

---

## 8. 诊断量（无训练信号）

每个 update 的第一个 mini-batch 上取一小撮观测（\(x_0=0\)）：

1. **路径直度。** 路径中点处网络速度相对 \(u^\star=x_0-x_1\) 的偏差，再除以路径长度（越小越直）。
2. **离散化间隙。** 对 \(k\in\{1,4,8,16\}\) 报告 \(\|a_N-a_k\|\)。

用于监控是否变直；不进入 \(\mathcal{L}\)。

---

## 9. 评测协议（与 Method 配套）

按下列协议报告平均回报：

| 协议 | 起点 | 步数 |
|------|------|------|
| zero@\(N\) | \(x_0=0\) | \(N\in\{64,32,16,8,4,1\}\) |
| random@\(N\) | \(x_0\sim\mathcal{N}(0,I)\) | 同上 |

Reflow 主主张：**零起点下少步相对满步的掉分很小**，且满步 zero 与 baseline 打平。随机起点满步是另一轴，由第 6 节处理。

---

## 10. Sim2Real：训练设定、域随机化与教师的作用

本节描述将 **Ours**（`all_ideas_teacher_kd`，含 PPO 教师 KD）落到真机时的仿真侧设定。环境与宇树官方 `Unitree-Go2-Velocity` MDP 对齐（45-D policy 观测、官方奖励与 PD 部署接口）；策略为条件整流流 Actor，部署时导出固定欧拉步 ONNX（真机主用 **1-step**）。

### 10.1 RL 策略与部署形态

- **训练算法。** 学生为 FPO++ 流策略，叠加 RA-reflow、少步→教师动作 KD、辅助零初值混合与自适应积分预算（见第 7 节）。Teacher 为冻结的官方 Gaussian PPO，只提供动作目标，不回传梯度；Reflow 端点仍由学生（或约定的流教师）积分，**不用 PPO 充当流场端点**。
- **采集。** 默认从 \(x_0\sim\mathcal{N}(0,I)\) 起步做满步积分；辅助项中以概率 \(p_0\) 将 \(x_0\) 置零，使 zero@1 与 random@\(N\) 同时可部署。
- **部署。** Obs 与训练一致（无 `base_lin_vel` 的 45-D）；动作经与训练相同的 scale / clip 后写关节目标。真机侧用 bake 好的 `ours_s1`（或 `ours_s64`）ONNX，控制频率与宇树 LowCmd 链路对齐（见 `go2_deploy`）。

### 10.2 域随机化与观测噪声

沿用官方 Go2 velocity 任务的 Event / Obs 配置（`velocity_env_cfg.EventCfg`），仿真→真机的主要随机化包括：

| 类别 | 设定（摘要） |
|------|----------------|
| 地面摩擦 | startup 随机静/动摩擦 \(\in[0.3,1.2]\)，restitution \(\in[0,0.15]\)，64 buckets |
| 基座质量 | startup 对 `base` 加性质量扰动 \(\in[-1,3]\,\mathrm{kg}\) |
| 外推 / 推扰 | interval 以 5–10 s 间隔施加平面速度推扰 \(v_x,v_y\in[-0.5,0.5]\) |
| Reset | 根位姿均匀扰动；关节速度扰动；外力项默认置零（保留接口） |
| 观测噪声 | \(\omega\)、投影重力、关节位置/速度上的加性均匀噪声（与官方 PPO 相同量级） |

速度指令在训练中按 curriculum / level 采样；真机评测时用记录的 `cmd_traj` 钉死仿真 `cmd_vel`，以便对齐同一指令下的关节轨迹（HW cmd replay）。

### 10.3 为什么 Teacher 在 Sim2Real 里仍然重要

少步部署（尤其 \(N=1\)）是真机实时性的刚需。仅 FPO++ 或仅 PPO-KD 时，play 上常出现 **满步尚可、1-step 崩溃**；带 PPO 教师的完整方法把学生少步动作拉向已可部署的 Gaussian 动作，使 **zero@1 ≈ zero@64**，从而能直接导出 1-step ONNX。

因此 Teacher 在 Sim2Real 链路中的角色不是「再涨一点仿真回报」，而是：

1. **补齐少步几何**：让流策略在极低欧拉预算下仍输出接近教师的可行动作；
2. **对齐可部署先验**：PPO 教师本身已在同一 MDP + 随机化下训成，蒸馏把「仿真可走」的动作流形传给学生；
3. **降低真机试错成本**：1-step 稳定后，可用同一 ONNX 做 HW cmd replay / 真机行走，而不必在机上再扫积分步数。

经验上，Go2 上 Ours@1 与 PPO 的跟踪误差同量级，而 FPO++@1 / kd_only@1 会明显发散或跌倒（见实验记录 §定量定性）。**没有教师时，流策略很难同时满足「仿真满步回报」与「真机 1-step 可控」**。

---

## 附录 A. 变体 ↔ 开关对照

| 名称 | Reflow | 模式 | EMA 端点 | 自适应 | Theory 日志 | \(\mathcal{L}_{\mathrm{rx0}}\) | 混合 \(x_0\) |
|------|--------|------|----------|--------|-------------|-------------------------------|--------------|
| `baseline` | 关 | — | — | 关 | 关 | 关 | 关（默认 random 采集） |
| `reflow` | 开 | uniform | 否 | 关 | 关 | 关 | 关 |
| `reward_aware` | 开 | reward_aware | 否 | 关 | 关 | 关 | 关 |
| `adaptive_compute` | 开 | uniform | 否 | 开 | 关 | 关 | 关 |
| `fpo_operator` | 开 | fpo_operator | 是 | 关 | 关 | 关 | 关 |
| `theory` | 开 | uniform | 否 | 关 | 开 | 关 | 关 |
| `all_ideas` | 开 | reward_aware | 否 | 开 | 开 | 关 | 关 |
| `all_ideas_fpo` | 开 | fpo_operator | 是 | 开 | 开 | 关 | 关 |
| `reflow_random_x0` | 开 | uniform | 否 | 关 | 关 | 开 | 开 |

模式互斥：`reward_aware` 与 `fpo_operator` 不能同时生效。

---

## 附录 B. 代码锚点

| 组件 | 位置 |
|------|------|
| Reflow / adaptive / rx0 损失 | `modules/actor_critic.py` |
| Advantage 权重、步数头、指标 | `modules/reflow_extensions.py` |
| 更新循环、EMA 端点副本 | `algorithms/fpo.py` |
| 超参 | `rl_cfg.py` |
| `--fpo_variant` 预设 | `task_cfgs.py` |

超参名：`reflow_loss_coef` \(\lambda_r\)，`reflow_n_samples_per_obs` \(n\)，`reflow_advantage_threshold` \(\tau\)，`adaptive_compute_loss_coef` \(\lambda_a\)，`adaptive_latency_penalty_coef` \(\alpha\)，`random_x0_consistency_coef` \(\lambda_x\)，`train_flow_x0_mode` / `train_flow_x0_random_prob`。

---

*与 `isaaclab_fpo` 在 2026-08-25 前后实现一致。若字段增删，以 `rl_cfg.py` / `task_cfgs.py` 为准。*
