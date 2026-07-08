# RoboBenchMart True SFT-Reference KL-PPO 与 PickToBasket Reward 修改详解

本文档解释 RoboBenchMart PickToBasket 在 RLinf 中做 PPO 后训练时，为什么需要补 **SFT-reference KL**，以及为什么要重新设计 PickToBasket 的 shaped reward。

目标是让后续复盘时能看懂四件事：

```text
1. 改了哪些文件。
2. 修改前是什么问题。
3. 修改后代码逻辑是什么。
4. 这些修改对 SFT -> PPO -> correction / RECAP-lite 技术路线有什么作用。
```

## 1. 背景：为什么 SFT 后不能直接长训 PPO

当前主线是：

```text
PickToBasket-only SFT
-> KL-PPO 闭环强化
-> 失败 episode 生成 correction 数据
-> RECAP-lite weighted SFT
-> train / robo / unseen / OOD 评测
```

SFT 之后，模型已经学到了一部分有用能力：

```text
看图像
理解 prompt
靠近目标物体
夹取目标物体
移动到篮子附近
尝试放置
```

PPO 的目标不是从零学习这些能力，而是在 SFT 能力基础上做小幅闭环改进。如果 PPO 更新太激进，或者 reward 不准确，模型很容易退化。

典型失败包括：

```text
reward 上升，但 success rate 下降
模型学会夹起来，但不放进篮子
模型把物体带到篮子附近，但没有稳定放置
模型在训练任务上变好，但 OOD 更差
模型离 SFT 初始策略越来越远
```

因此这个项目里的 PPO 必须满足两个条件：

```text
1. 策略更新要被 SFT reference 约束住。
2. reward 要尽量接近最终 success，而不是鼓励中间状态刷分。
```

对应的代码修改就是：

```text
true SFT-reference KL-PPO
PickToBasket event/progress shaped reward
```

## 2. 修改文件总览

本轮核心修改文件：

```text
rlinf/workers/actor/fsdp_actor_worker.py
examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml
examples/embodiment/config/rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05.yaml
rlinf/envs/robobenchmart/robobenchmart_env.py
```

各文件作用：

```text
rlinf/workers/actor/fsdp_actor_worker.py
```

在 embodied PPO actor 中保存一份 SFT 初始模型作为 reference policy，并在训练时计算 `ref_logprobs`，把 reference KL 加入 actor loss。

```text
examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml
examples/embodiment/config/rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05.yaml
```

把 RBM PPO 的 KL penalty 改成非负 `mse`，使它更像保守的距离惩罚。

```text
rlinf/envs/robobenchmart/robobenchmart_env.py
```

把 PickToBasket reward 从容易重复奖励中间状态的 per-step 状态奖励，改成事件型 + 进度型 reward。

## 3. PPO 训练里每个量从哪里来

你之前问过 PPO 需要这些量：

```text
action
logprob
value
old_logprob
ref_logprob
advantage
KL
```

下面逐个说明。

### 3.1 action

`action` 是 rollout policy 在环境中执行的动作。

在 RoboBenchMart PickToBasket 中，action 是连续动作，当前使用：

```text
action_dim = 13
```

OpenPI 通常一次预测一个 action chunk。对 PickToBasket 来说，动作不是单个 token，而是一段连续控制序列。

### 3.2 old_logprob

`old_logprob` 是 rollout 阶段产生 action 时，旧策略对这个 action 给出的 log probability。

PPO update 时会用：

```text
ratio = exp(current_logprob - old_logprob)
```

这个 ratio 用来衡量当前策略相对采样策略变化了多少。

PPO clip 约束的是：

```text
current policy vs rollout old policy
```

它能防止单次 update 太大，但不能保证多轮训练之后仍然接近最初 SFT 模型。

### 3.3 value

`value` 是 value head 对当前状态的回报估计。

它用于计算：

```text
return
advantage
value loss
```

如果 value 学得很差，advantage 会很噪，PPO 更新也会不稳定。

### 3.4 advantage

`advantage` 表示某个动作比 value 预期更好还是更差。

直观理解：

```text
advantage > 0：这个动作比预期好，PPO 应该提高它的概率
advantage < 0：这个动作比预期差，PPO 应该降低它的概率
```

在机器人任务里，advantage 的质量高度依赖 reward。如果 reward 鼓励了错误的中间行为，advantage 也会推动模型往错误方向走。

### 3.5 current logprob

PPO update 时，actor 会用当前模型重新计算同一批 action 的 log probability：

```text
current_logprob = current_policy(action | observation, prompt, diffusion inputs)
```

它和 `old_logprob` 一起构成 PPO ratio。

### 3.6 ref_logprob

`ref_logprob` 是这次修改新增的核心量。

它是冻结的 SFT 初始模型对同一批 action 的 log probability：

```text
ref_logprob = SFT_reference(action | same observation, same prompt, same diffusion inputs)
```

注意这里必须是同一批输入、同一批动作，否则 KL 没有意义。

### 3.7 KL

这里有两个不同的 KL 概念，必须区分。

```text
PPO approx_kl
```

比较的是：

```text
current policy vs rollout old policy
```

它用于观察 PPO 单次 update 相对采样策略动了多少。

```text
SFT reference KL
```

比较的是：

```text
current policy vs initial SFT policy
```

它用于防止 PPO 多轮训练后偏离 SFT 太远。

你的技术路线中真正需要的是第二个。

## 4. 修改前：为什么原 embodied PPO 还不是真正的 KL-PPO

修改前，`EmbodiedFSDPActor` 的训练流程大致是：

```text
1. rollout worker 采样 action，并保存 old_logprob / value / reward / forward_inputs
2. actor worker 根据 reward 和 value 计算 advantage / return
3. actor 用当前模型重新计算 current_logprob / value
4. 用 current_logprob 和 old_logprob 做 PPO clipped objective
5. 加 value loss / entropy loss 等项
6. backward 更新 actor
```

这个流程有 PPO clip，但没有显式 SFT reference policy。

也就是说修改前有：

```text
current_logprob vs old_logprob
```

没有：

```text
current_logprob vs ref_logprob
```

即使配置里写了类似：

```yaml
algorithm:
  kl_beta: 0.001
  kl_penalty: kl
```

在这条 embodied PPO 路径里，也没有真正变成：

```text
loss += kl_beta * KL(current, SFT_reference)
```

这就是问题所在。

PPO clip 只能限制一步更新，不限制长期漂移。举例：

```text
第 1 轮：policy_1 离 SFT 很近
第 2 轮：policy_2 离 policy_1 很近
第 3 轮：policy_3 离 policy_2 很近
...
第 30 轮：policy_30 可能已经离 SFT 很远
```

对于机器人策略，这种漂移很危险，因为 SFT 学到的基础行为可能被 reward 噪声破坏。

## 5. 修改后：Embodied PPO 加入 true SFT-reference KL

### 5.1 初始化时保存 SFT reference 权重

修改文件：

```text
rlinf/workers/actor/fsdp_actor_worker.py
```

在 `EmbodiedFSDPActor` 初始化和 worker 初始化阶段新增逻辑：

```text
如果 kl_beta > 0：
  保存当前 actor 模型权重到 CPU
  这份权重作为 frozen SFT reference policy
```

为什么当前 actor 初始化权重就是 SFT？

因为 PPO 启动时配置里使用的是 SFT PyTorch checkpoint：

```yaml
actor.model.model_path: ${MODEL_PATH}
rollout.model.model_path: ${MODEL_PATH}
```

所以 actor 初始化完成那一刻的权重，就是 PPO 要约束的 SFT 初始策略。

### 5.2 为什么 reference 权重放 CPU

OpenPI pi0.5 模型很大。如果额外在 GPU 上常驻一份完整 reference model，会显著增加显存占用。

这里采用的是：

```text
只保存 reference state_dict 到 CPU
训练 microbatch 时临时切换权重计算 ref_logprob
算完后切回当前训练权重
```

这样节省显存，但代价是 reference logprob 计算会慢一些。

### 5.3 为什么必须先算 ref_logprob，再算 current forward

最终实现顺序是：

```text
1. no_grad 下切到 SFT reference 权重
2. 计算 ref_logprobs
3. 切回当前训练权重
4. 用当前权重计算 current_logprobs / values
5. 计算 PPO loss + reference KL loss
6. backward
```

不能先 current forward 再切 reference 权重。原因是：如果 current forward 后还没有 backward，就修改模型权重，PyTorch autograd 可能认为参与计算图的 tensor 被 inplace 改动，触发错误。

因此正确顺序是：

```text
reference no_grad forward first
current train forward second
backward last
```

### 5.4 reference logprob 怎么算

新增逻辑相当于：

```python
with torch.no_grad():
    with cpu_weight_swap(model, ref_policy_state_dict, offload_model_buffer):
        model.eval()
        ref_output = model(
            forward_inputs=forward_inputs,
            compute_logprobs=True,
            compute_entropy=False,
            compute_values=False,
            use_cache=False,
        )
ref_logprobs = ref_output["logprobs"].detach()
```

关键点：

```text
forward_inputs 必须和 current policy 使用的是同一份。
```

因为 OpenPI action logprob 依赖的不只是 observation，还包括 diffusion 过程中的输入、denoise step、prompt token、图像、状态等信息。只有同一批 `forward_inputs` 下比较 `current_logprob` 和 `ref_logprob`，KL 才有意义。

### 5.5 reference KL loss 怎么算

修改后 loss 变成：

```text
loss = PPO actor/value loss + kl_beta * ref_kl_loss
```

其中：

```text
ref_kl_loss = penalty(current_logprob, ref_logprob)
```

当前 RBM PPO 使用的是：

```yaml
logprob_type: chunk_level
reward_type: chunk_level
```

因此 KL 会按 action chunk 聚合。

直观理解：

```text
如果当前策略对同一批 action 的概率分布和 SFT reference 差很多，loss 变大。
如果当前策略仍然接近 SFT reference，KL 惩罚较小。
```

### 5.6 新增 TensorBoard 指标

新增指标：

```text
train/actor/ref_kl_loss
train/actor/ref_kl_abs
```

含义：

```text
train/actor/ref_kl_loss
```

真正加入 loss 的 reference KL 项。它受 `kl_penalty` 类型影响。

```text
train/actor/ref_kl_abs
```

reference KL 的绝对值均值，用来观察 current policy 离 SFT reference 有多远。

它们和原来的：

```text
train/actor/approx_kl
```

不是同一个东西。

```text
approx_kl：current vs rollout old policy
ref_kl：current vs initial SFT policy
```

训练时应该同时看：

```text
approx_kl 是否爆炸
ref_kl 是否持续增大
success rate 是否真的提高
reward 是否和 success 同向
```

如果 reward 增大但 success 下降，说明 reward 可能被中间行为利用了。

## 6. 为什么 KL penalty 从 signed `kl` 改成 `mse`

修改文件：

```text
examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml
examples/embodiment/config/rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05.yaml
```

修改前：

```yaml
algorithm:
  kl_penalty: kl
```

在 RLinf 中，signed `kl` 近似形式类似：

```python
logprob - ref_logprob
```

它可能为正，也可能为负。

如果把 signed 值直接加到 loss：

```text
loss += kl_beta * (logprob - ref_logprob)
```

那么当这一项为负时，它会降低 loss，可能等价于鼓励某些偏离 reference 的方向。

当前阶段目标不是激进探索，而是保守约束 SFT，因此更适合使用非负距离。

修改后：

```yaml
algorithm:
  kl_penalty: mse
```

`mse` 形式类似：

```python
0.5 * (logprob - ref_logprob) ** 2
```

它始终非负，更适合当前用途：

```text
current logprob 越偏离 SFT reference logprob，惩罚越大。
```

这不是说 `mse` 永远比 signed KL 更正确，而是在当前小规模、保守、强 SFT 初始策略的机器人 PPO 阶段，它更符合目标。

## 7. PickToBasket reward 修改

修改文件：

```text
rlinf/envs/robobenchmart/robobenchmart_env.py
```

### 7.1 修改前 reward 的问题

旧 reward 更像 per-step 状态奖励：

```text
只要当前状态满足 lifted，就给 lifted reward
只要当前状态 near basket，就给 near-basket reward
只要当前状态 placed，就给 placed reward
```

这种设计的优点是 reward 密集，PPO 比较容易拿到非零信号。

但问题是，它容易让模型优化中间状态，而不是最终 success。

典型例子：

```text
模型夹起目标物体
模型把物体带到篮子附近
模型没有真正放进篮子
模型仍然每一步拿到 lifted / near-basket reward
最终官方 success = 0
```

这就解释了为什么会出现：

```text
视频里能夹起物体，但没有放进篮子
reward 变大，但成功率没有提高甚至下降
```

### 7.2 修改后 reward 结构

当前 reward 是：

```text
+5.00 * success
+1.00 * first_placed
+0.30 * first_lifted
+0.50 * positive_progress_to_basket
+0.30 * first_placed_static
-0.20 * first_non_target_displacement
```

代码层面还会 clamp：

```text
reward in [-0.2, 6.0]
```

防止单步 reward 过大或过小。

### 7.3 新增 episode 内状态缓存

为了实现“只奖励第一次事件”和“只奖励正向进展”，环境 wrapper 中保存了 episode 内状态：

```text
_target_lifted_once
_target_placed_once
_target_placed_static_once
_prev_basket_proximity
_non_target_penalty_applied
```

含义：

```text
_target_lifted_once
```

当前 episode 是否已经给过第一次夹起奖励。

```text
_target_placed_once
```

当前 episode 是否已经给过第一次放入篮子奖励。

```text
_target_placed_static_once
```

当前 episode 是否已经给过放入后稳定奖励。

```text
_prev_basket_proximity
```

历史最好的靠近篮子程度。只有比历史更好时，才给 progress reward。

```text
_non_target_penalty_applied
```

非目标物体扰动惩罚是否已经给过，避免每一步重复惩罚。

这些缓存必须在 reset 时清空。否则上一个 episode 的事件状态会污染下一个 episode。

### 7.4 每一项 reward 为什么这样设计

```text
+5.00 * success
```

最终成功奖励最大。PPO 应该优先优化官方 success，而不是中间动作。

```text
+1.00 * first_placed
```

第一次把目标物体放入篮子时给奖励。它比 `first_lifted` 大，因为“放进去”比“夹起来”更接近任务完成。

```text
+0.30 * first_lifted
```

第一次夹起目标物体时给少量奖励。它用于缓解 sparse reward，但权重不能太高，否则会鼓励“夹起来就停”。

```text
+0.50 * positive_progress_to_basket
```

只有目标物体比历史更接近篮子时才给奖励。这样可以防止模型停在篮子附近重复刷 reward。

```text
+0.30 * first_placed_static
```

目标物体放入后，机器人第一次进入稳定状态时给奖励。它鼓励稳定放置，而不是刚碰到篮子又掉出。

```text
-0.20 * first_non_target_displacement
```

第一次明显扰动非目标物体时给轻量惩罚。只罚一次，避免惩罚主导整个 episode。

### 7.5 为什么不是继续加大 lifted reward

你的视频里出现过“可以夹起来，但没放进篮子”。这说明模型已经能做到中间阶段，但缺的是后半段：

```text
移动到篮子上方
释放
稳定放入
完成 success
```

如果继续加大 `lifted`，PPO 更可能强化已经会的中间行为，而不是补齐后半段。

所以当前 reward 的方向是：

```text
lifted 给少量一次性奖励
placed / success 给更大权重
near basket 只奖励正向进展
```

## 8. 修改前后对比

### 8.1 KL-PPO 链路

修改前：

```text
PPO old_logprob clip 存在
没有 SFT reference policy
没有 ref_logprob
没有 ref_kl_loss
多轮 PPO 可能逐渐偏离 SFT
```

修改后：

```text
保存 SFT 初始权重为 reference
每个 microbatch 先算 ref_logprob
再算 current logprob / value
loss 中加入 kl_beta * ref_kl_loss
TensorBoard 记录 ref_kl_loss / ref_kl_abs
```

修改后的作用：

```text
PPO 可以探索，但会被拉回 SFT 附近。
如果 ref_kl 持续变大，说明策略正在偏离 SFT。
```

### 8.2 reward 设计

修改前：

```text
per-step lifted / near-basket / placed 状态奖励
reward 更密集
但容易奖励未完成任务的中间状态
```

修改后：

```text
事件型 first_lifted / first_placed / first_placed_static
进度型 positive_progress_to_basket
success 权重最大
减少重复刷中间状态 reward
```

修改后的作用：

```text
reward 更接近 success rate。
PPO 更难通过“夹起但不放入”刷高 reward。
```

代价：

```text
reward 会更稀疏。
PPO 可能需要更小 lr、更强 KL、更少 epoch、更严格 checkpoint selection。
```

## 9. 怎么判断训练是否正常

训练时不要只看 reward。至少看这些指标：

```text
success / eval_success
```

最终最重要。所有 PPO checkpoint 都必须和 matched SFT baseline 对比。

```text
reward
```

只能作为辅助。如果 reward 增大但 success 不增，说明 reward 可能仍然有偏差。

```text
train/actor/approx_kl
```

看 PPO 当前 update 是否太大。

```text
train/actor/ref_kl_loss
train/actor/ref_kl_abs
```

看当前策略是否越来越远离 SFT reference。

```text
value loss / explained variance
```

看 value 是否学到东西。如果 value 很差，advantage 会不可靠。

```text
clip fraction
```

如果长期很高，说明 PPO 更新太激进。

视频也必须看。机器人任务里，reward 曲线不能替代行为检查。

## 10. 当前实验结论

补上 true reference KL 后，训练链路更合理，但这不等于 PPO 已经带来性能提升。

当前观察是：

```text
matched SFT eval 仍然是更可靠的 baseline
旧 PPO 有 reward 上升但 success 下降的问题
true KL PPO 更保守，但仍没有稳定超过 SFT
直接长训 PPO 风险很高
```

所以当前推荐路线是：

```text
1. 保留 SFT 作为主 baseline。
2. PPO 只做 small-run probe 和失败数据生成。
3. 不把 Nestle/Slam/Duff 这类 OOD 目标直接放进 PPO 训练。
4. 使用 proxy tasks 提供训练分布扩展。
5. 从失败 episode 构造 correction 数据。
6. 做 RECAP-lite / weighted SFT。
7. 在 train / robo / unseen / OOD 上统一评测。
```

## 11. 一句话总结

这次修改解决两个核心问题：

```text
1. 让 embodied PPO 真正受到 SFT reference policy 约束，避免 PPO 训坏 SFT。
2. 让 PickToBasket reward 更接近最终 success，减少“夹起/靠近但不放入”的 reward hacking。
```

但它们只是让后训练链路更正确，不保证 PPO 一定超过 SFT。最终是否有效，必须以 matched eval success rate 和视频行为为准。
