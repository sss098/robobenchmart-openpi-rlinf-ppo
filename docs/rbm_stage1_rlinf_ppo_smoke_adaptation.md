# RoboBenchMart 阶段 1：RLinf PPO Smoke 适配说明

本文档记录本机 RLinf 仓库中为 RoboBenchMart `PickToBasket` 任务补充的阶段 1 适配。

目标不是马上训练出更强模型，而是先让下面这条强化学习闭环在代码上具备运行条件：

```text
RoboBenchMart env
-> RLinf rollout worker
-> OpenPI pi0.5 actor 采样 action
-> 同时得到 old_logprob 和 value
-> env.step(action) 得到 reward / done
-> RLinf 计算 advantage 和 KL-PPO loss
-> 更新 actor
-> 保存 checkpoint
```

这一步是后续 KL-PPO、失败 episode correction、RECAP-lite / weighted SFT 的基础。

---

## 1. 为什么需要这次适配

RLinf 原来已经支持 OpenPI + PPO，但主要面向 LIBERO、标准 ManiSkill、Robocasa 等环境。
RoboBenchMart 虽然也是基于 ManiSkill，但它的 observation、action、camera 名称和 reward 信息都和已有任务不完全一致。

主要差异：

| 项目 | LIBERO / 原 ManiSkill | RoboBenchMart |
|---|---|---|
| 第三视角图像 | `image` / `base_camera` | `left_base_camera_link/rgb` |
| 手腕图像 | `wrist_image` | `fetch_hand/rgb` |
| 额外视角 | 通常没有或命名不同 | `right_base_camera_link/rgb` |
| state | 常见 7/8 维 | Fetch qpos，15 维 |
| action | 常见 7 维末端动作 | `pd_joint_pos`，13 维 |
| success 信息 | 各环境不同 | `info["success"]` |
| prompt | task description | `env.language_instructions` |

如果不适配，RLinf 会把 RoboBenchMart 当成普通 ManiSkill 任务处理，容易出现：

- 图像 key 找不到；
- action 被错误转换成 7 维末端控制；
- policy transform 和 SFT 时不一致；
- PPO 中的 `old_logprob`、`value`、`advantage` 无法建立在正确输入上。

---

## 2. 这次补了哪些文件

### 2.1 RoboBenchMart policy transform

文件：

```text
/root/autodl-tmp/projects/RLinf/rlinf/models/embodiment/openpi/policies/robobenchmart_policy.py
```

新增两个类：

```python
RoboBenchMartInputs
RoboBenchMartOutputs
```

#### RoboBenchMartInputs 的作用

把 RoboBenchMart 的 observation 转成 OpenPI pi0.5 模型需要的输入。

输入字段：

```text
observation/image
observation/wrist_image
observation/extra_view_image
observation/state
prompt
actions
```

输出字段：

```text
state
image["base_0_rgb"]
image["left_wrist_0_rgb"]
image["right_wrist_0_rgb"]
image_mask
prompt
actions
```

映射关系：

| RoboBenchMart / RLinf key | OpenPI key |
|---|---|
| `observation/image` | `base_0_rgb` |
| `observation/wrist_image` | `left_wrist_0_rgb` |
| `observation/extra_view_image` | `right_wrist_0_rgb` |
| `observation/state` | `state` |
| `prompt` | `prompt` |

这和你之前 OpenPI SFT / eval 里的 RBM 适配保持一致。

#### RoboBenchMartOutputs 的作用

OpenPI 内部 action 通常 pad 到更高维度，RoboBenchMart 实际只需要前 13 维。

所以这里做：

```python
return {"actions": np.asarray(data["actions"][:, :13])}
```

这样环境收到的就是 `pd_joint_pos` 需要的 13 维 action。

---

### 2.2 RoboBenchMart OpenPI dataconfig

文件：

```text
/root/autodl-tmp/projects/RLinf/rlinf/models/embodiment/openpi/dataconfig/robobenchmart_dataconfig.py
```

新增类：

```python
LeRobotRoboBenchMartDataConfig
```

它负责告诉 OpenPI / RLinf：

1. LeRobot dataset 里的字段怎么改名成 policy 输入字段；
2. 训练和 rollout 时应该使用哪个 policy transform；
3. 输出 action 应该怎么还原给环境。

repack 映射：

```python
{
    "observation/image": "image",
    "observation/wrist_image": "wrist_image",
    "observation/extra_view_image": "extra_image",
    "observation/state": "state",
    "actions": "actions",
    "prompt": "prompt",
}
```

这里对应你 SFT 数据转换脚本生成的字段：

```text
image
wrist_image
extra_image
state
actions
task / prompt
```

---

### 2.3 注册 OpenPI 配置 `pi05_rbm`

文件：

```text
/root/autodl-tmp/projects/RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py
```

新增 import：

```python
from rlinf.models.embodiment.openpi.dataconfig.robobenchmart_dataconfig import (
    LeRobotRoboBenchMartDataConfig,
)
```

新增 config：

```python
TrainConfig(
    name="pi05_rbm",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
    ),
    data=LeRobotRoboBenchMartDataConfig(...),
    ...
)
```

作用：后续 RLinf config 可以写：

```yaml
actor:
  model:
    openpi:
      config_name: pi05_rbm
```

这样 actor / rollout / PPO 都会使用 RoboBenchMart 的数据转换逻辑。

---

### 2.4 RoboBenchMart env wrapper

文件：

```text
/root/autodl-tmp/projects/RLinf/rlinf/envs/robobenchmart/robobenchmart_env.py
/root/autodl-tmp/projects/RLinf/rlinf/envs/robobenchmart/__init__.py
```

新增类：

```python
RoboBenchMartEnv
```

它继承 RLinf 已有的：

```python
ManiskillEnv
```

但覆盖 RoboBenchMart 特有的 observation 和 reward。

#### observation 包装

RoboBenchMart raw obs：

```text
raw_obs["sensor_data"]["left_base_camera_link"]["rgb"]
raw_obs["sensor_data"]["fetch_hand"]["rgb"]
raw_obs["sensor_data"]["right_base_camera_link"]["rgb"]
raw_obs["agent"]["qpos"]
env.language_instructions
```

被包装成 RLinf rollout worker 需要的统一字段：

```python
{
    "main_images": left_base_camera_link,
    "wrist_images": fetch_hand,
    "extra_view_images": right_base_camera_link,
    "states": qpos,
    "task_descriptions": language_instructions,
}
```

这些字段会继续被 RLinf 的 OpenPI actor 处理成：

```text
observation/image
observation/wrist_image
observation/extra_view_image
observation/state
prompt
```

#### reward 包装

第一版 smoke 使用最保守的 sparse reward：

```text
success=True  -> reward = 1
success=False -> reward = 0
```

代码里读取：

```python
info["success"]
```

这足够用于 PPO smoke，后续如果需要更强学习信号，可以再加入 shaped reward，例如：

- 目标物体是否被移动；
- 是否进入 basket 附近；
- 是否碰乱非目标物体；
- 是否超时。

但第一阶段不建议先上复杂 reward，因为 reward shaping 很容易引入错误优化目标。

#### reset 行为

RoboBenchMart continuous env 通常需要：

```python
env.reset(options={"reconfigure": True})
```

所以 wrapper 默认在 reset 时补：

```python
options.setdefault("reconfigure", True)
```

---

### 2.5 注册新的 env type

文件：

```text
/root/autodl-tmp/projects/RLinf/rlinf/envs/__init__.py
```

新增：

```python
ROBOBENCHMART = "robobenchmart"
```

并在 `get_env_cls()` 中新增分支：

```python
elif env_type == SupportedEnvType.ROBOBENCHMART:
    from rlinf.envs.robobenchmart.robobenchmart_env import RoboBenchMartEnv
    return RoboBenchMartEnv
```

作用：yaml 里可以写：

```yaml
env_type: robobenchmart
```

RLinf 会自动创建 `RoboBenchMartEnv`。

---

### 2.6 RoboBenchMart action passthrough

文件：

```text
/root/autodl-tmp/projects/RLinf/rlinf/envs/action_utils.py
```

新增：

```python
def prepare_actions_for_robobenchmart(raw_chunk_actions, action_dim):
    return np.asarray(raw_chunk_actions, dtype=np.float32)[..., :action_dim]
```

并在 `prepare_actions()` 中新增：

```python
elif env_type == SupportedEnvType.ROBOBENCHMART:
    chunk_actions = prepare_actions_for_robobenchmart(...)
```

作用：RoboBenchMart 的 action 是 13 维 `pd_joint_pos`，不需要被转换成普通 ManiSkill 的 7 维末端动作。

如果不加这个分支，RLinf 可能会按普通 ManiSkill 逻辑把 action 拆成：

```text
world_vector
rotation_delta
gripper
```

这对 RBM 是错的。

---

### 2.7 pi05_rbm value head 兼容

文件：

```text
/root/autodl-tmp/projects/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py
```

修改：

```python
if self.config.config_name in ["pi05_maniskill", "pi05_libero", "pi05_rbm"]:
```

作用：让 `pi05_rbm` 和已有 `pi05_libero`、`pi05_maniskill` 一样使用 pi0.5 的 value head 设置。

PPO 需要 value，用于：

```text
delta_t = reward_t + gamma * V(s_{t+1}) - V(s_t)
advantage = GAE(delta_t)
```

如果没有 value head，就无法正常做 actor-critic PPO。

---

### 2.8 新增 env smoke 配置

文件：

```text
/root/autodl-tmp/projects/RLinf/examples/embodiment/config/env/robobenchmart_pick_to_basket_nivea.yaml
```

这是最小 RoboBenchMart 环境配置。

关键字段：

```yaml
env_type: robobenchmart
project_path: /root/autodl-tmp/projects/RoboBenchMart

init_params:
  id: PickToBasketContNiveaEnv
  config_dir_path: /root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket
  robot_uids: ds_fetch_basket
  control_mode: pd_joint_pos
  obs_mode: rgb
  sim_backend: cpu
```

第一版只接 `PickToBasketContNiveaEnv`，因为 smoke test 应该尽量窄。

---

### 2.9 新增 PPO smoke 总配置

文件：

```text
/root/autodl-tmp/projects/RLinf/examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml
```

用途：跑一个很小的 KL-PPO smoke。

关键设置：

```yaml
runner:
  max_epochs: 2
  save_interval: 1

env:
  train:
    total_num_envs: 4
    max_steps_per_rollout_epoch: 40

actor:
  micro_batch_size: 4
  global_batch_size: 16
  model:
    action_dim: 13
    num_action_chunks: 10
    add_value_head: True
    openpi:
      config_name: pi05_rbm
      noise_method: flow_noise
      joint_logprob: True
      value_after_vlm: True

algorithm:
  kl_beta: 0.001
  update_epoch: 1
  clip_ratio_high: 0.1
  clip_ratio_low: 0.1
```

这些参数很保守，目的是：

- 不把 SFT 模型训崩；
- 先确认 `old_logprob`、`value`、`advantage`、`KL` 都能产生；
- 先确认 checkpoint 能保存。

---

### 2.10 新增启动脚本

文件：

```text
/root/autodl-tmp/projects/RLinf/scripts_local/06_rbm_ppo_smoke.sh
```

用法：

```bash
MODEL_PATH=/path/to/rbm_pi05_sft_pytorch \
bash /root/autodl-tmp/projects/RLinf/scripts_local/06_rbm_ppo_smoke.sh
```

脚本会设置：

```bash
PYTHONPATH="$PROJECT:$RBM_PROJECT:$PYTHONPATH"
```

确保 RLinf 能 import RoboBenchMart 的 `dsynth.envs` 和 `dsynth.robots`。

---

## 3. PPO 中各个关键量来自哪里

### action

来自 rollout worker 中当前 actor：

```text
OpenPI pi0.5 actor.sample_actions(obs)
```

然后经过 RBM output transform：

```text
actions[:, :13]
```

送入 RoboBenchMart env。

### old_logprob

actor 在采样 action 时同时计算该 action 的 logprob。

在 RLinf 里对应：

```text
prev_logprobs
```

PPO update 时它就是：

```text
old_logprob
```

### value

来自 actor 上新增的 value head：

```yaml
actor.model.add_value_head: True
actor.model.openpi.value_after_vlm: True
```

RLinf rollout 会保存：

```text
prev_values
```

### ref_logprob

KL-PPO 需要冻结参考策略。

在完整 PPO 中，一般用 SFT checkpoint 作为 reference policy。
同一个 observation 和同一个 sampled action，会被 reference policy 重新计算 logprob。

阶段 1 smoke 的重点是先让 actor rollout 和 PPO buffer 通；后续需要进一步确认当前 RLinf 配置里的 KL/reference policy 路径是否已经按预期使用 SFT checkpoint 作为 ref。

### reward

当前 RBM wrapper 使用：

```text
success -> 1
not success -> 0
```

### advantage

由 RLinf PPO 根据 reward、done、value 自动计算。

配置：

```yaml
algorithm:
  adv_type: gae
  gamma: 0.99
  gae_lambda: 0.95
  normalize_advantages: True
```

GAE 公式：

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
```

### KL

KL 用来限制 PPO 不要偏离 SFT/reference policy 太远。

当前 smoke 配置：

```yaml
algorithm:
  kl_beta: 0.001
  kl_penalty: kl
  clip_ratio_high: 0.1
  clip_ratio_low: 0.1
```

完整 KL-PPO 时应重点监控：

```text
approx_kl
policy_loss
value_loss
clip_fraction
success_rate
```

---

## 4. 当前还不能直接正式 PPO 的原因

这次补的是阶段 1 代码适配，不等于已经能正式训练。

主要剩余问题：

### 4.1 需要 PyTorch 格式的 RBM SFT checkpoint

你的 20k SFT checkpoint 当前是 OpenPI/JAX Orbax 格式：

```text
/root/autodl-tmp/projects/openpi/checkpoints/pi05_sft_rbm_pick_to_basket/...
```

RLinf PPO 的 OpenPI actor 通常加载 PyTorch checkpoint。

smoke 配置里目前占位：

```text
/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch
```

下一步需要先完成：

```text
JAX/Orbax SFT checkpoint -> PyTorch OpenPI checkpoint
```

或者确认 RLinf 当前 OpenPI loader 能否直接加载你的 checkpoint 结构。

### 4.2 尚未实际启动 env + model smoke

已完成静态检查：

```text
python -m py_compile
bash -n
YAML safe_load
```

但没有真正启动 PPO，因为模型路径还没有准备好。

### 4.3 目前只接了 Nivea 一个任务

这是刻意的。

阶段 1 的目标是工程链路，不是成功率。
等 smoke 成功后，再扩展到：

```text
Fanta / Nivea / Stars
train / robo / unseen
```

---

## 5. 推荐下一步

### Step 1：准备 PyTorch SFT checkpoint

目标路径建议：

```text
/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch
```

要包含 RLinf OpenPI loader 需要的：

```text
model weights
assets / norm_stats
tokenizer / OpenPI assets
```

### Step 2：运行 PPO smoke

命令：

```bash
MODEL_PATH=/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch \
bash /root/autodl-tmp/projects/RLinf/scripts_local/06_rbm_ppo_smoke.sh
```

### Step 3：检查日志

重点看是否出现：

```text
prev_logprobs
prev_values
rewards
advantages
approx_kl / kl
policy_loss
value_loss
checkpoint save
```

如果缺 `prev_logprobs`，说明 OpenPI flow logprob 路径没通。

如果缺 `prev_values`，说明 value head 没接上。

如果 env 报 action shape 错误，优先检查：

```yaml
actor.model.action_dim: 13
actor.model.openpi.action_env_dim: 13
env_type: robobenchmart
```

### Step 4：只在 smoke 成功后扩大任务

建议顺序：

```text
1. Nivea, 4 env, 2 epochs
2. Nivea, 8-16 env, 5-10 updates
3. Fanta/Nivea/Stars train 混合
4. 加 robo randomized init
5. 加 unseen scenes
```

不要一开始就把完整 4 类评测和 PPO 训练混在一起。

---

## 6. 这次没有做什么

这次没有做：

- JAX SFT checkpoint 转 PyTorch；
- 正式 PPO 训练；
- RECAP 数据适配；
- failed episode correction 数据生成；
- 多任务混合采样；
- dense reward 设计；
- OOD item 训练。

这些是后续阶段。

---

## 7. 文件清单

本次阶段 1 相关文件：

```text
新增：
/root/autodl-tmp/projects/RLinf/rlinf/models/embodiment/openpi/policies/robobenchmart_policy.py
/root/autodl-tmp/projects/RLinf/rlinf/models/embodiment/openpi/dataconfig/robobenchmart_dataconfig.py
/root/autodl-tmp/projects/RLinf/rlinf/envs/robobenchmart/__init__.py
/root/autodl-tmp/projects/RLinf/rlinf/envs/robobenchmart/robobenchmart_env.py
/root/autodl-tmp/projects/RLinf/examples/embodiment/config/env/robobenchmart_pick_to_basket_nivea.yaml
/root/autodl-tmp/projects/RLinf/examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml
/root/autodl-tmp/projects/RLinf/scripts_local/06_rbm_ppo_smoke.sh

修改：
/root/autodl-tmp/projects/RLinf/rlinf/envs/__init__.py
/root/autodl-tmp/projects/RLinf/rlinf/envs/action_utils.py
/root/autodl-tmp/projects/RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py
/root/autodl-tmp/projects/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py
```

---

## 8. 验证记录

已执行并通过：

```bash
python -m py_compile \
  rlinf/envs/__init__.py \
  rlinf/envs/action_utils.py \
  rlinf/envs/robobenchmart/robobenchmart_env.py \
  rlinf/models/embodiment/openpi/policies/robobenchmart_policy.py \
  rlinf/models/embodiment/openpi/dataconfig/robobenchmart_dataconfig.py \
  rlinf/models/embodiment/openpi/dataconfig/__init__.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py
```

已执行并通过：

```bash
bash -n /root/autodl-tmp/projects/RLinf/scripts_local/06_rbm_ppo_smoke.sh
```

已执行并通过：

```text
YAML safe_load:
- examples/embodiment/config/env/robobenchmart_pick_to_basket_nivea.yaml
- examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml
```

尝试轻量 import `get_openpi_config("pi05_rbm")` 时进程被系统 kill。
这通常是 OpenPI/JAX/Torch import 过重导致的资源问题，不是语法错误。后续应在 GPU/内存充足时通过真实 smoke 进一步验证。


---

# 2026-07-04 更新：PickToBasket PPO 阶段扩展到 Train3 + Proxy Mix

本节记录在阶段 1 smoke 适配之后，对 PPO 阶段继续补充的内容。

当前目标不是把 `Nestle / Slam / Duff` 放进训练，而是在不泄露最终 OOD 评测目标的前提下，尽可能提高 PickToBasket 的泛化能力。

最终 PPO 阶段被拆成两条训练配置：

```text
阶段 2：Train3 KL-PPO small-run
训练：Fanta / Nivea / Stars
用途：先确认 PPO 能提高 in-domain，同时不明显损害 OOD

阶段 4：Train3 + Proxy mixed KL-PPO
训练：Fanta / Nivea / Stars + ProxyRandom
用途：加入非最终 OOD 商品，提高语言 grounding 和目标辨识泛化
```

最终 OOD 评测仍然保留：

```text
Nestle / Slam / Duff
```

它们不参与 PPO rollout，也不参与后续 correction 数据生成，除非明确把它们从 held-out OOD 改成训练可见任务。

---

## 10. 为什么要加入 proxy 任务

只用 `Fanta / Nivea / Stars` 做 SFT 和 PPO，最稳定提升的是 in-domain。但你的早期评测已经显示：

```text
SFT 后 train task 明显提升
OOD task 没有同步提升，甚至下降
```

这说明模型学到的可能是：

```text
见过商品 + 见过语言 prompt + 见过视觉外观
```

而不是更通用的：

```text
根据语言在货架上定位任意目标商品
```

因此 PPO 阶段如果继续只强化 `Fanta / Nivea / Stars`，很可能继续强化 in-domain 偏置。

proxy 任务的作用是加入“非最终 OOD 商品”，让模型在训练中见到更多商品外观和语言目标，但又不使用最终评测目标。

这类任务不是为了直接刷 proxy 成功率，而是为了给策略增加泛化压力：

```text
更多目标名称
更多外观
更多货架相邻干扰
更多抓取形状
更多语言 grounding 变化
```

---

## 11. Proxy 任务的设计原则

最终 OOD 目标不进入训练：

```text
Nestle Fitness Chocolate Cereals
SLAM luncheon meat
Duff Beer Can
```

原 train 目标也不作为 proxy：

```text
Fanta Sabor Naranja 2L
Nivea Body Milk
Nestle Honey Stars
```

另外排除了 `Vanish Stain Remover`，因为 RoboBenchMart 的原始配置注释中把它标为 test OOD 相关物体。

proxy 任务从当前 scene 中动态选择：

```text
当前 scene 里存在的商品
- train 目标
- held-out OOD 目标
- Vanish
= proxy 候选商品
```

这样做的原因是：`pick_to_basket_1` 和 `pick_to_basket_2` 的 inactive 商品并不完全相同。一开始只写死一组 proxy 商品时，测试发现某些 scene 里没有这些商品，导致 reset 报错。现在改成按当前 scene 动态采样后，两个布局组都能正常初始化。

---

## 12. 新增和修改的文件

### 12.1 新增 proxy env 注册文件

文件：

```text
/root/autodl-tmp/projects/RLinf/rlinf/envs/robobenchmart/proxy_tasks.py
```

新增：

```python
PickToBasketProxyRandomEnv
```

它继承：

```python
dsynth.envs.pick_to_basket.PickToBasketContEnv
```

核心逻辑：

```python
present_proxy_products = sorted(
    set(scene_products_df["product_name"].unique())
    - set(self.EXCLUDED_PRODUCT_NAMES)
)
```

然后每个 scene 用当前 scene 的随机数生成器采样一个 proxy 商品：

```python
product_name = self._batched_episode_rng[scene_idx].choice(
    present_proxy_products
)
```

这样 language instruction 会变成：

```text
move to shelf and pick Banania Cookies to basket
move to shelf and pick Monster Energy Drink to basket
...
```

排除列表：

```python
EXCLUDED_PRODUCT_NAMES = (
    "Fanta Sabor Naranja 2L",
    "Nivea Body Milk",
    "Nestle Honey Stars",
    "Nestle Fitness Chocolate Cereals",
    "SLAM luncheon meat",
    "Duff Beer Can",
    "Vanish Stain Remover",
)
```

用途：注册一个不会泄露最终 OOD 的 PickToBasket proxy 任务。

---

### 12.2 修改 RoboBenchMart wrapper，导入 proxy 注册

文件：

```text
/root/autodl-tmp/projects/RLinf/rlinf/envs/robobenchmart/robobenchmart_env.py
```

新增导入：

```python
import rlinf.envs.robobenchmart.proxy_tasks  # noqa: F401
```

作用：当 `RoboBenchMartEnv` 初始化时，自动注册 `PickToBasketProxyRandomEnv`。否则 `gym.make("PickToBasketProxyRandomEnv")` 会找不到这个 env id。

---

### 12.3 支持 `init_params.id` 列表

同一文件：

```text
/root/autodl-tmp/projects/RLinf/rlinf/envs/robobenchmart/robobenchmart_env.py
```

新增逻辑：

```python
def _select_robobenchmart_task(cfg, seed_offset: int):
    env_ids = cfg.init_params.id
    if not isinstance(env_ids, (list, tuple, ListConfig)):
        return cfg

    selected_env_id = str(env_ids[seed_offset % len(env_ids)])
    cfg = copy.deepcopy(cfg)
    with open_dict(cfg):
        cfg.init_params.id = selected_env_id
        cfg.selected_env_id = selected_env_id
    return cfg
```

作用：允许 YAML 里写多个 env id。每个 pipeline stage 根据 `seed_offset` 选择一个 env id。

例如：

```yaml
id:
  - PickToBasketContFantaEnv
  - PickToBasketContNiveaEnv
  - PickToBasketContStarsEnv
  - PickToBasketProxyRandomEnv
```

当：

```yaml
rollout.pipeline_stage_num: 4
```

对应关系是：

```text
stage 0 -> PickToBasketContFantaEnv
stage 1 -> PickToBasketContNiveaEnv
stage 2 -> PickToBasketContStarsEnv
stage 3 -> PickToBasketProxyRandomEnv
```

---

### 12.4 新增 proxy-only env 配置

文件：

```text
/root/autodl-tmp/projects/RLinf/examples/embodiment/config/env/robobenchmart_pick_to_basket_proxy.yaml
```

核心内容：

```yaml
init_params:
  id: PickToBasketProxyRandomEnv
  config_dir_path: /root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket
```

用途：单独测试 proxy env，或单独诊断 proxy reset、language instruction、RGB observation。

---

### 12.5 新增 Train3 + Proxy mix env 配置

文件：

```text
/root/autodl-tmp/projects/RLinf/examples/embodiment/config/env/robobenchmart_pick_to_basket_train_proxy_mix.yaml
```

核心内容：

```yaml
init_params:
  id:
    - PickToBasketContFantaEnv
    - PickToBasketContNiveaEnv
    - PickToBasketContStarsEnv
    - PickToBasketProxyRandomEnv
```

用途：混合 PPO rollout，约 75% train tasks + 25% proxy task。

为什么不是严格 70/30：RLinf 当前按 pipeline stage 建 env。用 4 stage 可以稳定实现 `Fanta/Nivea/Stars/ProxyRandom`。如果强行做 70/30，通常需要 10 stage，例如 7 train + 3 proxy，会显著增加环境实例和调度开销。因此当前采用更稳的工程折中：

```text
3 train stage + 1 proxy stage = 75/25
```

---

### 12.6 新增 mixed PPO 配置

文件：

```text
/root/autodl-tmp/projects/RLinf/examples/embodiment/config/rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05.yaml
```

相对 train3 PPO 配置，主要变化：

```yaml
defaults:
  - env/robobenchmart_pick_to_basket_train_proxy_mix@env.train
  - env/robobenchmart_pick_to_basket_train3@env.eval
```

训练并行：

```yaml
env:
  train:
    total_num_envs: 16

rollout:
  pipeline_stage_num: 4
```

含义：

```text
4 个 pipeline stage
每个 stage 4 个 env
总共 16 个 env
```

stage 分配：

```text
4 env -> Fanta
4 env -> Nivea
4 env -> Stars
4 env -> ProxyRandom
```

注意：

```text
runner.val_check_interval 默认仍为 -1。
```

也就是说，训练中默认不跑 RLinf 内部 eval。建议后续用单独评测脚本分别评测 train / OOD，避免 mixed PPO 的 `pipeline_stage_num=4` 和 train3/OOD2 的任务数量不一致导致评测采样不均。

---

### 12.7 新增启动脚本

文件：

```text
/root/autodl-tmp/projects/RLinf/scripts_local/07_rbm_ppo_proxy_mix.sh
```

作用：启动 Train3 + Proxy mixed PPO。

默认模型路径：

```text
/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch
```

运行：

```bash
MODEL_PATH=/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch \
bash /root/autodl-tmp/projects/RLinf/scripts_local/07_rbm_ppo_proxy_mix.sh
```

---

### 12.8 拆分 OOD 配置

原来尝试把下面三个任务放在同一个 OOD3 配置里：

```text
Nestle / Slam / Duff
```

测试后发现：

```text
test_unseen_items_pick_to_basket 里有 Nestle / Slam
但没有 Duff

demo_envs/pick_to_basket 里有 Duff
但不是 Nestle/Slam 的 unseen-items 目录
```

因此拆成两个配置：

```text
/root/autodl-tmp/projects/RLinf/examples/embodiment/config/env/robobenchmart_pick_to_basket_ood_items2.yaml
/root/autodl-tmp/projects/RLinf/examples/embodiment/config/env/robobenchmart_pick_to_basket_duff.yaml
```

`ood_items2`：

```yaml
id:
  - PickToBasketContNestleEnv
  - PickToBasketContSlamEnv
config_dir_path: /root/autodl-tmp/projects/RoboBenchMart/demo_envs/test_unseen_items_pick_to_basket
```

`duff`：

```yaml
id: PickToBasketContDuffEnv
config_dir_path: /root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket
```

这样避免 `gym.make/reset` 时出现：

```text
RuntimeError: Product Duff Beer Can is not present on scene #0
```

---

## 13. 已完成的测试

### 13.1 YAML / Python / shell 静态检查

已通过：

```text
yaml OK examples/embodiment/config/env/robobenchmart_pick_to_basket_proxy.yaml
yaml OK examples/embodiment/config/env/robobenchmart_pick_to_basket_train_proxy_mix.yaml
yaml OK examples/embodiment/config/env/robobenchmart_pick_to_basket_ood_items2.yaml
yaml OK examples/embodiment/config/env/robobenchmart_pick_to_basket_duff.yaml
yaml OK examples/embodiment/config/rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05.yaml
```

并通过：

```bash
python -m py_compile \
  rlinf/envs/robobenchmart/robobenchmart_env.py \
  rlinf/envs/robobenchmart/proxy_tasks.py

bash -n scripts_local/07_rbm_ppo_proxy_mix.sh
```

---

### 13.2 Train3 + Proxy mix 实际初始化测试

测试内容：

```text
stage 0 -> Fanta
stage 1 -> Nivea
stage 2 -> Stars
stage 3 -> ProxyRandom
```

每个 stage 都实际执行：

```text
gym.make(...)
env.reset(...)
observation 包装
language instruction 读取
RGB image shape 检查
```

通过输出：

```text
stage=0 env_id=PickToBasketContFantaEnv instruction=move to shelf and pick Fanta Sabor Naranja 2L to basket image_shape=(1, 256, 256, 3)
stage=1 env_id=PickToBasketContNiveaEnv instruction=move to shelf and pick Nivea Body Milk to basket image_shape=(1, 256, 256, 3)
stage=2 env_id=PickToBasketContStarsEnv instruction=move to shelf and pick Nestle Honey Stars to basket image_shape=(1, 256, 256, 3)
stage=3 env_id=PickToBasketProxyRandomEnv instruction=move to shelf and pick Banania Cookies to basket image_shape=(1, 256, 256, 3)
```

说明：

```text
proxy env 注册成功
proxy target 采样成功
language instruction 正常
RGB observation 正常
```

---

### 13.3 OOD 配置实际初始化测试

`Nestle / Slam` 在 `test_unseen_items_pick_to_basket` 中测试通过：

```text
stage=0 env_id=PickToBasketContNestleEnv instruction=move to shelf and pick Nestle Fitness Chocolate Cereals to basket image_shape=(1, 256, 256, 3)
stage=1 env_id=PickToBasketContSlamEnv instruction=move to shelf and pick SLAM luncheon meat to basket image_shape=(1, 256, 256, 3)
```

`Duff` 在 `demo_envs/pick_to_basket` 中单独测试通过：

```text
env_id=PickToBasketContDuffEnv instruction=move to shelf and pick Duff Beer Can to basket image_shape=(1, 256, 256, 3)
```

---

## 14. 当前推荐的 PPO 技术路线

### 阶段 1：PPO smoke

目标：确认 JAX SFT -> PyTorch checkpoint -> RLinf PPO 闭环能跑。

已完成：

```text
action rollout
old_logprob / logprob
value head
advantage / return
KL-PPO loss
checkpoint 保存
```

---

### 阶段 2：Train3 KL-PPO small-run

训练：

```text
PickToBasketContFantaEnv
PickToBasketContNiveaEnv
PickToBasketContStarsEnv
```

不训练：

```text
Nestle
Slam
Duff
```

目标：

```text
train success 提升
OOD 不明显下降
KL 不爆
value loss 稳定
```

推荐先跑：

```text
12 env
pipeline_stage_num = 3
horizon = 300
10 step probe
```

再跑：

```text
30 step small-run
```

---

### 阶段 3：Proxy 环境准备

已完成：

```text
PickToBasketProxyRandomEnv
robobenchmart_pick_to_basket_proxy.yaml
robobenchmart_pick_to_basket_train_proxy_mix.yaml
```

proxy 不包含：

```text
Fanta / Nivea / Stars
Nestle / Slam / Duff
Vanish
```

---

### 阶段 4：Train3 + Proxy mixed KL-PPO

训练：

```text
Fanta
Nivea
Stars
ProxyRandom
```

比例：

```text
75% train tasks
25% proxy task
```

目标：在不使用最终 OOD 目标的情况下，提高目标识别、语言 grounding 和商品外观泛化能力。

默认配置：

```text
rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05
```

默认并行：

```text
16 env
4 pipeline stages
每 stage 4 env
```

---

### 阶段 5：失败 episode correction

属于 RECAP-lite 前的数据生成阶段。

从阶段 2 / 阶段 4 的 rollout 中收集失败 episode，但不要收集最终 OOD 的训练 correction。

优先保留：

```text
找错目标
抓错相邻物体
目标找对但抓取失败
抓到目标但放篮失败
```

低价值或丢弃：

```text
完全乱走
没有接近货架
明显仿真异常
```

---

### 阶段 6：RECAP-lite weighted SFT

使用 correction 数据做 weighted SFT。

建议权重：

```text
目标 grounding 错误：高
抓错相邻物体：高
抓取失败：中高
放篮失败：中
完全无效 episode：低或丢弃
```

---

### 阶段 7：最终评测

至少比较：

```text
official pi0.5
20k SFT
Train3 PPO
Train3 + Proxy PPO
Train3 + Proxy PPO + RECAP-lite
```

任务：

```text
Train/in-domain:
- Fanta
- Nivea
- Stars

OOD / held-out:
- Nestle
- Slam
- Duff
```

注意：

```text
Duff 当前不在 test_unseen_items_pick_to_basket 目录中。
如果把 Duff 当作 OOD target，需要用单独的 Duff 配置评测。
```

---

## 15. 推荐运行命令

### 15.1 Train3 PPO probe

```bash
cd /root/autodl-tmp/projects/RLinf
source .venv_openpi/bin/activate

export PYTHONPATH=/root/autodl-tmp/projects/RLinf:/root/autodl-tmp/projects/RoboBenchMart:/root/autodl-tmp/projects/openpi:${PYTHONPATH:-}
export MODEL_PATH=/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch

bash examples/embodiment/run_embodiment.sh \
  rbm_pick_to_basket_ppo_openpi_pi05 \
  LIBERO \
  actor.model.model_path="$MODEL_PATH" \
  rollout.model.model_path="$MODEL_PATH" \
  runner.experiment_name=rbm_pick_to_basket_train3_ppo_probe10 \
  runner.max_epochs=10 \
  runner.save_interval=5 \
  runner.val_check_interval=-1 \
  env.train.total_num_envs=12 \
  env.eval.total_num_envs=12 \
  env.train.max_steps_per_rollout_epoch=300 \
  env.eval.max_steps_per_rollout_epoch=300 \
  rollout.pipeline_stage_num=3 \
  algorithm.update_epoch=1 \
  algorithm.kl_beta=0.001 \
  actor.optim.lr=1.0e-6 \
  actor.optim.value_lr=5.0e-5
```

---

### 15.2 Train3 + Proxy mixed PPO probe

```bash
cd /root/autodl-tmp/projects/RLinf
source .venv_openpi/bin/activate

export PYTHONPATH=/root/autodl-tmp/projects/RLinf:/root/autodl-tmp/projects/RoboBenchMart:/root/autodl-tmp/projects/openpi:${PYTHONPATH:-}
export MODEL_PATH=/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch

bash examples/embodiment/run_embodiment.sh \
  rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05 \
  LIBERO \
  actor.model.model_path="$MODEL_PATH" \
  rollout.model.model_path="$MODEL_PATH" \
  runner.experiment_name=rbm_pick_to_basket_proxy_mix_ppo_probe10 \
  runner.max_epochs=10 \
  runner.save_interval=5 \
  runner.val_check_interval=-1 \
  env.train.total_num_envs=16 \
  env.train.max_steps_per_rollout_epoch=300 \
  rollout.pipeline_stage_num=4 \
  algorithm.update_epoch=1 \
  algorithm.kl_beta=0.001 \
  actor.optim.lr=1.0e-6 \
  actor.optim.value_lr=5.0e-5
```

也可以用本地脚本：

```bash
MODEL_PATH=/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch \
bash /root/autodl-tmp/projects/RLinf/scripts_local/07_rbm_ppo_proxy_mix.sh \
  runner.experiment_name=rbm_pick_to_basket_proxy_mix_ppo_probe10 \
  runner.max_epochs=10 \
  runner.save_interval=5 \
  runner.val_check_interval=-1 \
  env.train.max_steps_per_rollout_epoch=300
```

---

## 16. 训练时应该重点看哪些指标

PPO 阶段重点看：

```text
rollout/rewards
env/num_trajectories
rollout/advantages_mean
rollout/advantages_max
rollout/returns_mean
train/actor/approx_kl
train/actor/ratio
train/actor/clip_fraction
train/actor/grad_norm
train/critic/value_loss
train/critic/explained_variance
```

如果长期出现：

```text
rollout/rewards = 0
env/num_trajectories = 0
```

不要盲目加 epoch。优先考虑：

```text
把 horizon 从 300 增加到 500
检查 success reward 是否过稀疏
先收集失败 episode 做 correction
```

如果出现：

```text
approx_kl 快速变大
ratio 明显偏离 1
grad_norm 异常升高
OOD success 明显下降
```

应该停止当前 PPO run，回退到更保守配置：

```text
降低 actor lr
提高 kl_beta
减少 update_epoch
缩短训练步数
```

---

## 17. 2026-07-05 追加：官方 seed 对齐、动作轨迹对比、PPO shaped reward

这次追加解决三个问题：

1. RLinf eval 的 episode seed 要继续对齐 RoboBenchMart 官方 JAX eval。
2. 能在同一个 seed 下比较 JAX server 和 RLinf PyTorch 实际下发到环境的 action。
3. PPO 训练前补 shaped reward，避免“已经夹起但没有放进篮子”仍然只有 0 reward。

### 17.1 官方 JAX eval 的 seed 递增逻辑

官方脚本 `RoboBenchMart/scripts/eval_policy_client.py` 的逻辑是：

```python
seed = start_seed - 1
init_pose_seed = robot_init_pose_start_seed - 1

for traj_idx in range(num_traj):
    seed += 1
    init_pose_seed += 1
    env.reset(seed=seed, options={
        "reconfigure": True,
        "robot_init_pose_seed": init_pose_seed,
    })
```

所以如果命令是：

```text
--start-seed 0
--robot-init-pose-start-seed 10000
-n 3
```

则三条轨迹是：

| trajectory | scene seed | robot init pose seed |
|---|---:|---:|
| 0 | 0 | 10000 |
| 1 | 1 | 10001 |
| 2 | 2 | 10002 |

RLinf 之前只固定了 wrapper 的初始 `seed`，没有显式表达“每条 eval episode 递增”。现在在：

```text
/root/autodl-tmp/projects/RLinf/rlinf/envs/robobenchmart/robobenchmart_env.py
```

新增了：

```python
episode_seed_start: Optional[int]
robot_init_pose_start_seed: Optional[int]
```

当 `episode_seed_start=0` 时，full reset 会按 episode 生成：

```text
0, 1, 2, ...
```

当 `robot_init_pose_start_seed=10000` 时，full reset 会按 episode 生成：

```text
10000, 10001, 10002, ...
```

注意：这里没有再按 pipeline stage 加 offset。原因是你原来的官方 eval 通常是分别评测 Fanta / Nivea / Stars，每个任务都从同一个 `start_seed` 开始。RLinf 三 stage 并行评测时，如果 stage 0/1/2 分别代表 Fanta/Nivea/Stars，那么每个 stage 也应该各自跑 `0,1,2...`，这样才等价于官方逐任务评测。

相关配置已经加入所有 RBM env YAML：

```yaml
seed: 0
episode_seed_start: null
robot_init_pose_start_seed: null
```

PPO 主配置中 eval 默认设置为：

```yaml
env:
  eval:
    reward_mode: only_success
    episode_seed_start: 0
    robot_init_pose_start_seed: 10000
```

### 17.2 JAX server vs RLinf PyTorch action trace 对比

官方 JAX eval 下发 action 前会做：

```python
action = action.astype(np.float32)
action[8] = 0
action[9] = 0
```

RLinf 之前已经在：

```text
/root/autodl-tmp/projects/RLinf/rlinf/envs/action_utils.py
```

补了 `prepare_actions_for_robobenchmart`，同样会：

```python
chunk_actions = raw_actions[..., :13].astype(np.float32)
chunk_actions[..., 8] = 0.0
chunk_actions[..., 9] = 0.0
```

这次又在：

```text
/root/autodl-tmp/projects/RLinf/rlinf/workers/env/env_worker.py
```

新增了 eval action trace 导出：

```yaml
env.eval.action_trace_dir: /path/to/trace_dir
```

启用后，RLinf 每个 eval chunk 会保存一个 `.npz`：

```text
eval_stage0_rank0_chunk000000.npz
eval_stage1_rank0_chunk000000.npz
eval_stage2_rank0_chunk000000.npz
...
```

里面的 `actions` 是已经经过 RoboBenchMart action 后处理、真正要送进环境的动作。

新增两个脚本：

```text
/root/autodl-tmp/projects/RLinf/scripts_local/08_rbm_collect_jax_action_trace.py
/root/autodl-tmp/projects/RLinf/scripts_local/09_compare_rbm_action_traces.py
```

#### 采集 JAX server trace

先启动你的官方 OpenPI JAX policy server，然后运行：

```bash
cd /root/autodl-tmp/projects/RLinf
source .venv_openpi/bin/activate

export PYTHONPATH=/root/autodl-tmp/projects/RLinf:/root/autodl-tmp/projects/RoboBenchMart:/root/autodl-tmp/projects/openpi:${PYTHONPATH:-}

python scripts_local/08_rbm_collect_jax_action_trace.py \
  --env-id PickToBasketContFantaEnv \
  --scene-dir /root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket \
  --host localhost \
  --port 8000 \
  --start-seed 0 \
  --robot-init-pose-start-seed 10000 \
  --num-traj 3 \
  --max-horizon 600 \
  --output /root/autodl-tmp/projects/RLinf/logs/action_traces/jax_fanta_seed0_3.npz
```

#### 采集 RLinf PyTorch trace

```bash
cd /root/autodl-tmp/projects/RLinf
source .venv_openpi/bin/activate

export PYTHONPATH=/root/autodl-tmp/projects/RLinf:/root/autodl-tmp/projects/RoboBenchMart:/root/autodl-tmp/projects/openpi:${PYTHONPATH:-}
export MODEL_PATH=/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch
export OMP_NUM_THREADS=1

rm -rf /root/autodl-tmp/projects/RLinf/logs/action_traces/rlinf_train3_seed0_3
mkdir -p /root/autodl-tmp/projects/RLinf/logs/action_traces/rlinf_train3_seed0_3

bash examples/embodiment/run_embodiment.sh \
  rbm_pick_to_basket_ppo_openpi_pi05 \
  LIBERO \
  actor.model.model_path="$MODEL_PATH" \
  rollout.model.model_path="$MODEL_PATH" \
  runner.logger.experiment_name=rbm_pick_to_basket_trace_eval \
  runner.only_eval=True \
  runner.max_epochs=1 \
  runner.save_interval=-1 \
  runner.val_check_interval=-1 \
  algorithm.eval_rollout_epoch=3 \
  algorithm.sampling_params.do_sample=False \
  algorithm.sampling_params.temperature_eval=0.0 \
  env.eval.total_num_envs=3 \
  env.eval.max_steps_per_rollout_epoch=600 \
  env.eval.auto_reset=True \
  env.eval.ignore_terminations=True \
  env.eval.episode_seed_start=0 \
  env.eval.robot_init_pose_start_seed=10000 \
  env.eval.action_trace_dir=/root/autodl-tmp/projects/RLinf/logs/action_traces/rlinf_train3_seed0_3 \
  env.eval.video_cfg.save_video=False \
  rollout.pipeline_stage_num=3
```

这条命令的 `n` 是 3 条 trajectory 总量：3 个 stage，每个 stage 1 个 env，`eval_rollout_epoch=3`，所以每个任务会得到 3 条 episode。

如果只想和 Fanta 的 JAX trace 比较，取 RLinf 的 stage 0 文件目录；如果要逐任务比较：

| stage | 任务 |
|---:|---|
| 0 | `PickToBasketContFantaEnv` |
| 1 | `PickToBasketContNiveaEnv` |
| 2 | `PickToBasketContStarsEnv` |

#### 比较 trace

```bash
cd /root/autodl-tmp/projects/RLinf
source .venv_openpi/bin/activate

python scripts_local/09_compare_rbm_action_traces.py \
  /root/autodl-tmp/projects/RLinf/logs/action_traces/jax_fanta_seed0_3.npz \
  /root/autodl-tmp/projects/RLinf/logs/action_traces/rlinf_train3_seed0_3 \
  --stage-id 0 \
  --atol 1e-4
```

输出会包含：

```text
compared_shape=...
mean_abs=...
max_abs=...
rmse=...
per_dim_mean_abs=...
allclose=True/False
first_mismatch_idx=...
```

如果比较 Fanta，用 `--stage-id 0`；比较 Nivea 用 `--stage-id 1`；比较 Stars 用 `--stage-id 2`。如果 `mean_abs/max_abs` 很大，说明 PyTorch 转换模型和 JAX server 的动作分布没有对齐，需要优先排查模型转换、normalization、OpenPI transform、采样参数。

### 17.3 PPO 前补 shaped reward

原来的 reward 默认是：

```python
reward = info["success"].float()
```

这对 eval 是正确的，因为最终指标就是 official success。
但对 PPO 训练太稀疏：

- 夹起目标物但没放进篮子：0
- 目标物已经靠近篮子但没完全满足 success：0
- 快成功但机器人还没静止：0

所以现在在 `RoboBenchMartEnv` 中新增：

```yaml
reward_mode: rbm_shaped
```

训练 reward 变成：

```text
1.00 * success
+0.35 * is_obj_placed
+0.20 * lifted_target_object
+0.25 * target_object_near_basket
-0.50 * non_target_object_displaced
```

并 clamp 到：

```text
[-0.5, 1.8]
```

各项含义：

| 项 | 作用 |
|---|---|
| `success` | 保持最终目标最大权重 |
| `is_obj_placed` | 目标物进篮子附近，即使机器人还未静止，也给奖励 |
| `lifted_target_object` | 目标物相对初始高度抬高超过 3cm，鼓励先学会抓取 |
| `target_object_near_basket` | 目标物越接近机器人篮子，奖励越高 |
| `non_target_object_displaced` | 撞掉非目标物直接惩罚 |

注意：eval 仍然必须使用：

```yaml
reward_mode: only_success
```

否则评测 reward 就不是官方 success 指标了。

当前 PPO 配置已经设置为：

```yaml
env:
  train:
    reward_mode: rbm_shaped
  eval:
    reward_mode: only_success
```

### 17.4 修改文件清单

本次追加修改/新增：

```text
rlinf/envs/robobenchmart/robobenchmart_env.py
```

作用：

- 新增 `episode_seed_start`，对齐官方 scene seed 递增。
- 调整 `robot_init_pose_start_seed`，按 episode 递增。
- 新增 `reward_mode=rbm_shaped`。
- shaped reward 使用目标物抬升、目标物靠近篮子、入篮、非目标物扰动信息。

```text
rlinf/workers/env/env_worker.py
```

作用：

- 新增 `env.eval.action_trace_dir` 导出 eval action chunk。
- trace 保存的是后处理后实际送进环境的 action。

```text
examples/embodiment/config/env/robobenchmart_pick_to_basket*.yaml
```

作用：

- 新增 `episode_seed_start: null`。
- 新增 `action_trace_dir: null`。

```text
examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml
examples/embodiment/config/rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05.yaml
```

作用：

- PPO train 默认启用 `reward_mode=rbm_shaped`。
- PPO eval 默认保持 `reward_mode=only_success`。
- eval 默认使用 `episode_seed_start=0` 和 `robot_init_pose_start_seed=10000`。

```text
scripts_local/08_rbm_collect_jax_action_trace.py
scripts_local/09_compare_rbm_action_traces.py
```

作用：

- 从官方 JAX websocket server 采集同 seed action trace。
- 比较 JAX trace 和 RLinf PyTorch trace 的动作差异。

### 17.5 已完成的验证

已通过：

```bash
python -m py_compile \
  rlinf/envs/robobenchmart/robobenchmart_env.py \
  rlinf/workers/env/env_worker.py \
  scripts_local/08_rbm_collect_jax_action_trace.py \
  scripts_local/09_compare_rbm_action_traces.py
```

已通过 YAML 解析检查：

```text
env yaml ok
ppo yaml ok
```

已通过脚本语法检查：

```bash
bash -n scripts_local/06_rbm_ppo_smoke.sh scripts_local/07_rbm_ppo_proxy_mix.sh
```

我在当前工具进程里尝试真实 env smoke 时，环境报：

```text
RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver
```

这是当前工具进程看不到 Vulkan/GPU renderer，不是本次 Python/YAML 修改的语法错误。你之前的 shell 可以跑 RoboBenchMart 视频 eval，所以真实 smoke 需要在你那个能访问 Vulkan 的 shell 里执行。

### 17.6 下一步建议

先不要直接长训。先做两步小验证：

1. 跑 deterministic eval + action trace，确认 RLinf PyTorch 和 JAX server 同 seed 动作差异是否可接受。
2. 跑 1 epoch PPO smoke，确认 `rbm_shaped` reward 不再全是 0，并观察 tensorboard 里的 reward / success_once / KL / value loss。

如果 action trace 差异很大，优先修模型转换或 normalization；如果 action trace 接近但 success 仍低，才进入 PPO shaped reward 训练。
