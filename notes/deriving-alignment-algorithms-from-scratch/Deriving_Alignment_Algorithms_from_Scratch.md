# Deriving Alignment Algorithms from Scratch
### A First-Principles Journey through Preference Alignment

**Author:** Akivian ([@Akivian](https://github.com/Akivian)) · **Repository:** [github.com/Akivian/Principia-AI](https://github.com/Akivian/Principia-AI) · **Date:** October 2026 · **Version:** v1.0

---

> **$Q$：在已知人类相对偏好数据 $(x, y_w \succ y_l)$ 的情况下，如何以简洁直接的方式，训练模型生成更受偏好的回答？**

---

## 记号与约定

### 1. 样本空间与数据
* **$x \sim \mathcal{D}$**：输入提示词，服从提示词分布 $\mathcal{D}$；
* **$y \in \mathcal{Y}$**：模型生成序列，$\mathcal{Y}$ 为所有候选输出序列构成的离散集合；
* **$\mathcal{D}_{\text{pref}} = \left\{ (x^{(i)}, y_w^{(i)}, y_l^{(i)}) \right\}_{i=1}^N$**：成对偏好数据集。对每个提示词 $x$，$y_w$ 为人类更偏好的序列，$y_l$ 为相对落选的序列。

### 2. 策略模型
* **$\pi(y \mid x)$**：给定提示词 $x$ 时生成序列 $y$ 的条件分布，满足：$\forall x, \ \sum_{y \in \mathcal{Y}} \pi(y \mid x) = 1$；
* **$\pi_\theta$**：待优化的参数化策略（$\theta$ 为可训练参数）；
* **$\pi_{\text{ref}}$**：固定参考策略（多为监督微调后的初始模型），训练中参数冻结。

### 3. 辅助函数与常数
* **$r(x, y) \in \mathbb{R}$**：潜在标量奖励，衡量在提示词 $x$ 下生成 $y$ 的质量；
* **$\beta > 0$**：正则化强度，控制策略相对 $\pi_{\text{ref}}$ 的偏离程度；
* **$\sigma(z) = \frac{1}{1 + e^{-z}}$**：标准 Logistic 函数（Sigmoid）。

---

## 建模假设

**Bradley–Terry 偏好模型**：假设存在不可观测的真实奖励 $r^*(x, y)$，人类偏好服从该模型。 
给定提示词 $x$，序列 $y_w$ 优于 $y_l$ 的概率定义为二者上的**二项 Softmax**（等价于 Sigmoid）：
$$P(y_w \succ y_l \mid x) \triangleq \frac{\exp(r^*(x, y_w))}{\exp(r^*(x, y_w)) + \exp(r^*(x, y_l))}$$

分子分母同除以 $\exp(r^*(x, y_w))$：
$$P(y_w \succ y_l \mid x) = \frac{1}{1 + \exp\big(-(r^*(x, y_w) - r^*(x, y_l))\big)} = \sigma\big(r^*(x, y_w) - r^*(x, y_l)\big)$$

---

## 预备引理

### 引理 1：吉布斯不等式（KL 散度非负性）

对任意离散概率分布 $P, Q \in \Delta(\mathcal{Y})$（满足 $\sum_y P(y) = \sum_y Q(y) = 1$）：
* **信息熵**：$H(P) \triangleq -\sum_{y \in \mathcal{Y}} P(y) \ln P(y)$（系统的平均信息量）；
* **交叉熵**：$H(P, Q) \triangleq -\sum_{y \in \mathcal{Y}} P(y) \ln Q(y)$（误用分布 $Q$ 描述真实分布 $P$ 时的平均代价）；
* **KL 散度（相对熵）**：
  $$\mathbb{D}_{\text{KL}}(P \parallel Q) \triangleq H(P, Q) - H(P)$$
  $$= \sum_{y \in \mathcal{Y}} P(y) \ln \frac{P(y)}{Q(y)} \ge 0$$

> **【命题】** 对任意离散概率分布 $P, Q \in \Delta(\mathcal{Y})$，其 KL 散度 $\mathbb{D}_{\text{KL}}(P \parallel Q) \triangleq \sum_y P(y) \ln \frac{P(y)}{Q(y)} \ge 0$，等号成立当且仅当 $\forall y, P(y) = Q(y)$。

**【证明】** 
切线放缩：$\forall t > 0, \ \ln t \le t - 1$ 
原不等式等价于证：$\sum_{y} P(y) \ln \frac{Q(y)}{P(y)} \le 0$ 
由切线放缩：
$$\ln\left(\frac{Q(y)}{P(y)}\right) \le \frac{Q(y)}{P(y)} - 1$$
两边同乘 $P(y) > 0$：
$$P(y) \ln\left(\frac{Q(y)}{P(y)}\right) \le P(y) \left(\frac{Q(y)}{P(y)} - 1\right) = Q(y) - P(y)$$
全域求和：
$$\therefore \sum_{y \in \mathcal{Y}} P(y) \ln\left(\frac{Q(y)}{P(y)}\right) \le \sum_{y \in \mathcal{Y}} \big( Q(y) - P(y) \big) = \sum_{y \in \mathcal{Y}} Q(y) - \sum_{y \in \mathcal{Y}} P(y) = 1 - 1 = 0$$
$$\therefore -\sum_{y \in \mathcal{Y}} P(y) \ln\frac{P(y)}{Q(y)} \le 0$$
故：
$$\sum_{y \in \mathcal{Y}} P(y) \ln\frac{P(y)}{Q(y)} \ge 0 \iff \mathbb{D}_{\text{KL}} \ge 0$$
由切线放缩取等条件知：当且仅当 $\forall y, \frac{Q(y)}{P(y)} = 1$，即 $Q(y) = P(y)$ 时取等。 $\blacksquare$

---

### 引理 2：KL 散度约束下的最优策略（吉布斯变分形式）

> **【命题】** 目标泛函 $\mathcal{J}(\pi) \triangleq \sum_y \pi(y \mid x) r(x, y) - \beta \, \mathbb{D}_{\text{KL}}\big(\pi(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x)\big)$ 在约束 $\sum_y \pi(y \mid x) = 1$ 下，存在全局最优解：
> $$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
> 其中归一化函数 $Z(x) \triangleq \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$。

**【证明】** 
代入 KL 散度定义：
$$\mathcal{J}(\pi) = \sum_y \pi(y \mid x) r(x, y) - \beta \sum_y \pi(y \mid x) \ln \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$
$$= -\beta \left[ \sum_y \pi(y \mid x) \ln \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \frac{1}{\beta} \sum_y \pi(y \mid x) r(x, y) \right]$$
$$= -\beta \left[ \sum_y \pi(y \mid x) \left( \ln \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \frac{1}{\beta} r(x, y) \right) \right]$$
$$= -\beta \sum_y \pi(y \mid x) \left[ \ln \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \ln\left( \exp\left( \frac{1}{\beta} r(x, y) \right) \right) \right]$$
$$= -\beta \sum_y \pi(y \mid x) \ln \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)}$$

上式已接近 $\sum P(y) \ln \frac{P(y)}{Q(y)}$ 的 KL 散度形式，但分母尚未归一化。为直接调用引理 1 刻画极大值点，对分母进行配凑使之满足 $\sum Q(y) = 1$： 
定义分母各项之和为 $Z(x) \triangleq \sum_{y'} \pi_{\text{ref}}(y' \mid x) \exp\left( \frac{1}{\beta} r(x, y') \right)$； 
定义归一化后合法概率分布 $\pi^*(y \mid x) \triangleq \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$； 
$$\therefore \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right) = Z(x) \cdot \pi^*(y \mid x)$$

代入上式：
$$\mathcal{J}(\pi) = -\beta \sum_y \pi(y \mid x) \ln \frac{\pi(y \mid x)}{Z(x) \cdot \pi^*(y \mid x)}$$
$$= -\beta \sum_y \pi(y \mid x) \left[ \ln \frac{\pi(y \mid x)}{\pi^*(y \mid x)} - \ln Z(x) \right]$$
$$= -\beta \sum_y \pi(y \mid x) \ln \frac{\pi(y \mid x)}{\pi^*(y \mid x)} + \beta \sum_y \pi(y \mid x) \ln Z(x)$$
$$= -\beta \sum_y \pi(y \mid x) \ln \frac{\pi(y \mid x)}{\pi^*(y \mid x)} + \beta \ln Z(x) \underbrace{\sum_y \pi(y \mid x)}_{= 1}$$
$$\therefore \mathcal{J}(\pi) = -\beta \, \mathbb{D}_{\text{KL}}\big(\pi(\cdot \mid x) \parallel \pi^*(\cdot \mid x)\big) + \beta \ln Z(x) \quad \text{（常数项）}$$

由引理 1，$\mathbb{D}_{\text{KL}}\big(\pi(\cdot \mid x) \parallel \pi^*(\cdot \mid x)\big) \ge 0$。  
$$\therefore \text{当且仅当 } \pi(y \mid x) = \pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right) \text{ 时，达到 } \mathcal{J}(\pi) \text{ 取最大值 } \beta \ln Z(x) \quad \blacksquare$$

---

## 1. 出发点

### 1.1 无约束期望奖励最大化
设输入提示词 $x \sim \mathcal{D}$，候选生成序列 $y \in \mathcal{Y}$，参数化语言模型为条件概率分布 $\pi_\theta(\cdot \mid x) \in \Delta(\mathcal{Y})$，其中 $\Delta(\mathcal{Y})$ 表示离散概率单纯形。 
设存在潜在标量奖励函数 $r(x, y) \in \mathbb{R}$。一个自然的优化目标是最大化策略在整个数据分布上的期望奖励：
$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi(\cdot \mid x)} [r(x, y)] \right] = \max_\pi \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{y \in \mathcal{Y}} \pi(y \mid x) r(x, y) \right]$$

$\because x$ 之间相互独立，全局期望最大化可逐步解耦为单样本优化问题：
$$\forall x \sim \mathcal{D}, \quad \max_{\pi(\cdot \mid x) \in \Delta(\mathcal{Y})} \sum_{y \in \mathcal{Y}} \pi(y \mid x) r(x, y) \quad \text{s.t.} \ \pi(y \mid x) \ge 0, \ \sum_{y \in \mathcal{Y}} \pi(y \mid x) = 1$$

**然而，若直接用基础策略梯度（$\nabla_\theta \mathcal{J}_0 = \mathbb{E}[r \nabla_\theta \ln \pi_\theta]$）优化上述无约束目标，非常容易碰到：**
* **Reward Hacking**
* **Mode Collapse**：过度集中于少数高奖励模式

在传统任务中，奖励函数多由奖励模型拟合，天然存在漏洞，故该思路实践意义有限；**然而，若把任务限制在可验证奖励范畴，用具有客观对错标准的形式化验证器替换 $r(x, y)$，也可以得到有趣的结果，例如 DeepSeek-R1-Zero**：
* 完全不依赖人类思维链数据（零监督微调），且不对参考策略施加 KL 散度约束（$\beta = 0$）；
* 直接在基座模型上，在去除 KL 散度惩罚（$\beta = 0$）的设定下，用组内相对优势做无约束强化学习；
* 随训练推进自发出现“长思考”（思维标记显著增多），并表现出类似元认知的自检行为：
  > *e.g. "Wait, wait, wait. That's not right. Let me check again..."*
* 并可能以非人类习惯进行探索。

但该模型的问题也同样显而易见，并不意外：推理过程格式混乱，可读性低，容易陷入无限推理，且输出多语言混杂。

---

## 2. 正则约束下的在线优化

### 2.1 引入 KL 散度约束
为防止策略在最大化奖励时过度偏离，引入相对参考策略 $\pi_{\text{ref}}(\cdot \mid x)$ 的 KL 散度作为罚项，得到带正则的目标：
$$\max_{\theta \in \mathbb{R}^d} \mathcal{J}(\theta; x) \triangleq \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) r(x, y) - \beta \, \mathbb{D}_{\text{KL}}\big(\pi_\theta(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x)\big)$$
其中，$\pi_\theta(\cdot \mid x) \in \Delta(\mathcal{Y})$ 为由参数 $\theta$ 控制的策略分布，$\Delta(\mathcal{Y})$ 为定义在离散序列空间 $\mathcal{Y}$ 上的概率单纯形，$\beta > 0$ 为正则化系数。

代入离散 KL 散度定义 $\mathbb{D}_{\text{KL}}(P \parallel Q) \triangleq \sum_y P(y) \ln \frac{P(y)}{Q(y)}$，展开目标函数：
$$\mathcal{J}(\theta; x) = \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) r(x, y) - \beta \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) \ln \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

注意到任务奖励项与 KL 散度惩罚项均以当前策略 $\pi_\theta(y \mid x)$ 作为概率测度，由数学期望线性性质，两项可直接合并为同一期望算子下的标量形式：
$$\mathcal{J}(\theta; x) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ r(x, y) - \beta \ln \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] \triangleq \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} [R_\theta(x, y)]$$
其中，$R_\theta(x, y)$ 为融合了 KL 散度正则的序列级奖励函数，定义为：
$$R_\theta(x, y) \triangleq r(x, y) - \beta \ln \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

该合并显著降低采样开销。仅需单次在线自回归采样得到序列 $y \sim \pi_\theta$，即可完成两项计算。

**然而，由于离散采样导致梯度断裂：**
1. **采样算子不可微**：文本生成 $y \sim \pi_\theta$ 属于离散随机采样，该操作本身不可导，离散 token 无法把梯度回传到连续参数 $\theta$；
2. **期望算子与求导算子不可交换**：期望 $\mathbb{E}_{y \sim \pi_\theta}[f(y)]$ 的积分测度是 $\pi_\theta$ 本身，而该分布依赖参数 $\theta$。因此一般不能把 $\nabla_\theta$ 直接移入期望内部，即
  $$\nabla_\theta \mathbb{E}_{y \sim \pi_\theta}[f(y)] \ne \mathbb{E}_{y \sim \pi_\theta}[\nabla_\theta f(y)]$$
  策略梯度定理正是为处理这一难点：将对参数的导数改写为对数概率加权形式，使梯度可在采样期望下估计。

---

### 2.2 策略梯度的推导
为求解 $\nabla_\theta \mathcal{J}(\theta; x)$，展开求和式并应用乘积求导法则：
$$\nabla_\theta \mathcal{J}(\theta; x) = \nabla_\theta \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) R_\theta(x, y) = \sum_{y \in \mathcal{Y}} \big( \nabla_\theta \pi_\theta(y \mid x) \big) R_\theta(x, y) + \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) \big( \nabla_\theta R_\theta(x, y) \big)$$

展开后半求和项：
$$\sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) \nabla_\theta \left( r(x, y) - \beta \ln \pi_\theta(y \mid x) + \beta \ln \pi_{\text{ref}}(y \mid x) \right)$$
$\because r(x, y) + \beta \ln \pi_{\text{ref}}(y \mid x)$ 均与 $\theta$ 无关
$$\therefore = \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) \cdot \big( -\beta \nabla_\theta \ln \pi_\theta(y \mid x) \big) = -\beta \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) \cdot \nabla_\theta \ln \pi_\theta(y \mid x)$$
又 $\because \nabla_\theta \ln f(\theta) = \frac{1}{f(\theta)} \cdot \nabla_\theta f(\theta)$
$$= -\beta \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) \cdot \frac{\nabla_\theta \pi_\theta(y \mid x)}{\pi_\theta(y \mid x)} = -\beta \sum_{y \in \mathcal{Y}} \nabla_\theta \pi_\theta(y \mid x) = -\beta \nabla_\theta \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x)$$
由全概率归一化：
$$= -\beta \nabla_\theta(1) = -\beta \cdot 0 \equiv \mathbf{0}$$

$\therefore$ 后半项由于全概率归一化而严格恒等于 0。 
对前半项，应用对数导数恒等式 $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \ln \pi_\theta$：
$$\nabla_\theta \mathcal{J}(\theta; x) = \sum_{y \in \mathcal{Y}} \big( \nabla_\theta \pi_\theta(y \mid x) \big) R_\theta(x, y)$$
$$= \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) \, R_\theta(x, y) \, \nabla_\theta \ln \pi_\theta(y \mid x)$$
$$= \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ R_\theta(x, y) \, \nabla_\theta \ln \pi_\theta(y \mid x) \right]$$

文本生成中 $\mathcal{Y}$ 搜索空间极大、无法遍历，故仅能使用蒙特卡洛采样。由上式，策略梯度已化为采样期望；对每次采得的 $y$，只需计算标量回报 $R_\theta(x, y)$ 与对数概率梯度，即可对该期望做无偏估计。

---

### 2.3 减小方差
**然而，直接用蒙特卡洛抽样估计上述梯度，在离散文本空间中方差极大。** 
为此，引入与具体生成动作 $y$ 无关的状态基线 $V(x)$。基线项的期望梯度严格恒为 0：
$$\mathbb{E}_{y \sim \pi_\theta} \left[ V(x) \nabla_\theta \ln \pi_\theta(y \mid x) \right] = V(x) \sum_{y \in \mathcal{Y}} \nabla_\theta \pi_\theta(y \mid x) = V(x) \nabla_\theta (1) \equiv 0$$

若不减去基线，回报尺度往往整体偏正，优势函数偏弱的样本也会被同向强化，估计方差更大；减去与动作无关的基线后，优势函数更接近零均值，随机梯度更新更稳。按最小二乘意义，使梯度估计方差最小的基线，正是当前策略下的期望回报：
$$V^*(x) \triangleq \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} [R_\theta(x, y)]$$

于是策略梯度可无偏地改写为以**优势函数**为权重的形式：
$$\nabla_\theta \mathcal{J}(\theta; x) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ A(x, y) \nabla_\theta \ln \pi_\theta(y \mid x) \right]$$
$$A(x, y) \triangleq R_\theta(x, y) - V(x)$$

**然而，核心问题随即变为：如何构造 $V(x)$？** 
要精确计算 $V^*(x)$，理论上需遍历 $\mathcal{Y}$ 中全部候选回答（规模约为 $|\mathcal{V}|^L$，极度冗余），并在训练中持续更新，实践中不可行。如何高效、低偏差地估计未知期望回报 $V^*(x)$，便成为在线强化学习的核心问题。

---

### 2.4 路线 A：价值网络与多步优势估计
面对高维空间中未知的期望回报 $V^*(x)$，一个自然想法是用参数化的**状态价值函数**去逼近。兼顾表征能力与计算量，可引入与策略网络规模相当的独立**价值网络** $V_\phi(s_t)$，用以拟合状态价值函数（即条件期望回报）：
$$V^{\pi_\theta}(s_t) \triangleq \mathbb{E}_{\pi_\theta} \left[ \sum_{l=0}^\infty \gamma^l R_{t+l} \;\middle|\; s_t \right]$$

在 2.3 的推导中，方差最小的最优基线是当前策略下的期望回报。若把任务奖励记为 $R_{\text{task}}$（KL 项并入下文逐步奖励），则样本级最优基线可写为
$$b^*(x) \triangleq \mathbb{E}_{y \sim \pi_\theta}[R_{\text{task}}(x, y)]$$
**然而，这在长序列生成中显然会带来归因困难：**
* *例如*：一条含 $T$ 个 token 的序列前 $T-1$ 步都正确，仅第 $T$ 步算错，环境给出零奖励 $R_{\text{task}}(x, y) = 0$。在整句标量基线下，整条序列的优势函数值均为负：
  $$0 - b(x) < 0$$
  该全局负信号会一并惩罚前 $T-1$ 步正确的局部决策，反之亦然。
* 同时，状态空间随长度 $T$ 指数增长，完整轨迹的蒙特卡洛回报方差随长度累积，宏观基线 $b(x)$ 难以有效抑制轨迹内部 token 级扰动带来的方差。

#### 马尔可夫决策过程建模
为定位导致最终成败的具体决策步，把文本生成写成离散时间马尔可夫决策过程：
* **状态空间 $\mathcal{S}$**：至当前时刻的全部前缀 $s_t = (x, y_{<t})$，起始状态为 $s_1 = (x)$；
* **动作空间 $\mathcal{A}$**：词表中的下一 token $a_t = y_t \in \mathcal{V}$；
* **状态转移 $P$**：确定性拼接 $s_{t+1} = (s_t \circ a_t)$；序列结束时引入虚拟终止状态 $s_{T+1}$，并约定状态价值函数满足 $V(s_{T+1}) = 0$；
* **单步奖励 $R_t$**：在常见的“显式奖励 + 在线强化学习”设定中，中间步只承担 token 级分布偏离惩罚，终止步 $t=T$ 结算任务奖励 $R_{\text{task}}(x, y)$，并取折现因子 $\gamma = 1$：
  $$R_t(s_t, a_t) = \begin{cases} -\beta \ln \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)}, & 1 \le t < T \\ R_{\text{task}}(x, y) - \beta \ln \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)}, & t = T \end{cases}$$

映射至该决策过程后，局部决策是否合理，便由**动作价值函数**与**状态价值函数**之差——**优势函数**——来衡量：
$$A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)$$
由于动作空间就是词表，该优势函数天然定义在 token 粒度上。

估计优势函数时，若直接用从当前时刻累加的经验回报 $G_t^{\text{MC}} \triangleq \sum_{l=0}^{T-t} \gamma^l R_{t+l}$，轨迹一长方差便过高；若只靠价值网络对状态价值函数做单步自举：
$$\hat{Q}(s_t, a_t) = R_t + \gamma V_{\phi,\text{old}}(s_{t+1})$$
虽能降低未来随机性带来的方差，却把价值网络对状态价值函数的逼近误差带进梯度。 
定义单步时序差分残差为：
$$\delta_t \triangleq R_t + \gamma V_{\phi,\text{old}}(s_{t+1}) - V_{\phi,\text{old}}(s_t), \quad t \in \{1, 2, \dots, T\}$$
由 $V(s_{T+1}) = 0$，终止步残差为 $\delta_T = R_T - V_{\phi,\text{old}}(s_T)$。

#### 偏差与方差的折中
为在“蒙特卡洛：高方差、低偏差”与“单步自举：低方差、高偏差”之间取得可控折中，把未来各步**时序差分残差**做指数加权求和，得到优势函数的**多步估计**。对长度为 $T$ 的有限轨迹，约定终止后 $\delta_t = 0$（$t > T$），故
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} \triangleq \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$
* 当 $\lambda = 0$ 时，$\hat{A}_t^{\text{GAE}(\gamma, 0)} = \delta_t$，仅依赖单步自举，方差最低，但对价值网络拟合状态价值函数时的偏差敏感；
* 当 $\lambda = 1$ 时，展开求和可得：
  $$\hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^{T-t} \gamma^l R_{t+l} + \gamma^{T-t+1} V_{\phi,\text{old}}(s_{T+1}) - V_{\phi,\text{old}}(s_t) = G_t^{\text{MC}} - V_{\phi,\text{old}}(s_t)$$
  此时退化为基于完整轨迹的蒙特卡洛优势函数估计：无偏，但方差随序列变长而显著增大。

得到上述优势函数估计后，下一步是用它更新策略网络与价值网络。  
由策略梯度定理，更新应朝优势函数增大的方向进行。但训练样本来自旧策略 $\pi_{\text{old}}$，与当前 $\pi_\theta$ 有偏离，故引入重要性采样比
$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$
并以截断限制更新幅度，得到策略侧目标：
$$\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\text{old}}} \left[ \min\left( \rho_t(\theta) \hat{A}_t^{\text{GAE}}, \, \text{clip}\big(\rho_t(\theta), 1-\epsilon, 1+\epsilon\big) \hat{A}_t^{\text{GAE}} \right) \right]$$
同时，价值网络将该优势函数估计与旧的状态价值函数估计合成目标回报 $\hat{G}_t \triangleq \hat{A}_t^{\text{GAE}} + V_{\phi,\text{old}}(s_t)$，用均方误差回归状态价值函数：
$$\mathcal{L}_{V}(\phi) = \mathbb{E}_{s_t} \left[ \big( V_\phi(s_t) - \hat{G}_t \big)^2 \right]$$

至此，针对 2.3 的基线估计问题，我们借助决策过程建模与价值网络，并在偏差—方差之间做权衡，给出了一条用参数化价值网络逼近状态价值函数的可行路径。

**然而，在长思维链、大规模推理等场景下，该方案是否仍“高效且低偏差”？是否真正回答了 2.3 的核心问题？**

上述以价值网络（逼近状态价值函数）与多步优势估计为核心的框架，在超大规模推理与长思维链面前有明显瓶颈：
* **系统开销与显存压力**：与策略网络规模相当的价值网络，再加上奖励模型与参考模型，带来额外显存与跨节点通信压力，制约上下文长度与并发；
* **环境往往不提供中间过程的密集信号**（例如仅终局可验证的任务）；
* **价值网络自身难收敛**：目标分布快速变化时，状态价值函数拟合困难，贝尔曼自举还会把误差逐步放大，价值函数估计可能崩溃。

---

### 2.5 路线 B：组内相对基线
回到 2.3 的结论：使方差最小的最优基线，就是当前采样分布下的期望回报：
$$b^*(x) = \mathbb{E}_{y \sim \pi_\theta} [R_{\text{task}}(x, y)]$$

既然用参数化价值网络估计 token 级状态价值函数 $V(s_t)$ 会带来归因困难，并付出显存与自举偏差的代价，一个更朴素的统计思路是：**不再跨分布去拟合全局期望；对给定提示词 $x$，在局部条件下采一组输出，用组内经验均值直接当该样本的基线，从而彻底去掉价值网络。**

对给定提示词 $x$，用当前旧策略 $\pi_{\text{old}}$ 独立采出 $G$ 条候选轨迹：
$$\mathcal{Y}_G = \{y_1, y_2, \dots, y_G\}, \quad y_i \sim \pi_{\text{old}}(\cdot \mid x)$$
再经规则验证器（面向数学、代码等可验证任务）给每条完整回答打标量任务奖励，得到 $\boldsymbol{R} = [R_1, R_2, \dots, R_G]^T$，其中 $R_i \triangleq R_{\text{task}}(x, y_i)$。

该提示词 $x$ 在当前旧策略分布下的期望回报，由组内样本均值给出蒙特卡洛估计：
$$\text{mean}(\boldsymbol{R}) = \frac{1}{G} \sum_{i=1}^G R_i$$
同时计算组内样本标准差，以衡量当前问题下采样轨迹离散程度：
$$\text{std}(\boldsymbol{R}) = \sqrt{ \frac{1}{G} \sum_{i=1}^G \big(R_i - \text{mean}(\boldsymbol{R})\big)^2 + \epsilon_{\text{std}} }$$
其中 $\epsilon_{\text{std}} > 0$（通常取 $10^{-6} \sim 10^{-4}$）用于维持数值稳定，防止组内奖励一致时除以零。

得到 $G$ 条采样输出及其绝对奖励 $\{R_1, R_2, \dots, R_G\}$ 后，问题变为：如何用这些相对奖励更新策略？  
直接最大化 $\mathbb{E}[R]$ 会像 2.3 一样：方差过大、训练震荡；不同提示词的奖励基线也相差很大，难以在统一尺度下稳定学习。  
于是仍用组内均值 $\text{mean}(\boldsymbol{R})$ 作基准，得到相对奖惩 $R_i - \text{mean}(\boldsymbol{R})$。  
但这尚未消除组内尺度差异——若某提示词组内波动极大，其梯度会主导参数更新。再除以组内标准差 $\text{std}(\boldsymbol{R})$，得到组相对优势：
$$\tilde{A}_i = \frac{R_i - \text{mean}(\boldsymbol{R})}{\text{std}(\boldsymbol{R})}$$

**然而，$\tilde{A}_i$ 是序列级标量，而策略梯度作用在 token 级（需计算 $\nabla_\theta \ln \pi_\theta(y_{i,t} \mid x, y_{i,<t})$）。**  
在 2.4 节，我们用价值网络为每个中间 token 估计状态价值函数 $V(s_t)$，再结合多步优势估计做细粒度归因；  
**本节既已去掉价值网络，便失去 token 级状态价值函数的估计能力，因而做出妥协：把整条序列的组相对优势 $\tilde{A}_i$ 平权广播到该回答的每一个生成 token：**
$$\hat{A}_{i,t} = \tilde{A}_i, \quad \forall t \in \{1, \dots, |y_i|\}$$
以 token 级归因精度的损失，换取去掉价值网络后的显存收益，并消除自举偏差。

为把 token 级优势函数写成可优化目标，并复用旧策略 $\pi_{\text{old}}$ 的采样，引入 token 级重要性采样比：
$$\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid x, y_{i,<t})}$$
得到朴素代理目标 $\mathbb{E}\big[\rho_{i,t}(\theta) \tilde{A}_i\big]$。  
然而，比率过大会引发梯度震荡，故沿用 2.4 节的截断约束：
$$\min\left( \rho_{i,t}(\theta) \tilde{A}_i, \, \text{clip}\big(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon\big) \tilde{A}_i \right)$$

同样需要 KL 散度正则，防止偏离参考模型 $\pi_{\text{ref}}$。  
注意：在 2.1～2.4 中，KL 散度被写进决策过程的单步奖励再做多步优势估计；价值网络必须拟合“含 KL 散度约束”的状态价值函数（总回报），否则价值函数估计与真实目标（任务奖励 + KL 散度）错位，优势函数估计失效。  
这加重了价值网络的拟合负担与自举误差：长序列上 KL 散度累积会带来巨大方差与长度偏差，冲淡任务奖励。  
**本节既无价值网络，也无单步递推与多步优势估计，便不必再把 KL 散度混进单步奖励。**  
于是把它挪到损失外层，作为独立正则项，避免不同轨迹长度下的累加污染组内标准差 $\text{std}(\boldsymbol{R})$，保证 $\tilde{A}_i$ 只反映任务相对优劣。

为削弱“越长序列累加项越多、越主导更新”的偏差，对序列内做长度归一化 $\frac{1}{|y_i|} \sum_{t=1}^{|y_i|}$，再在组内取平均，得到完整目标：
$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x, \{y_i\}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left\{ \min\left( \rho_{i,t}(\theta) \tilde{A}_i, \, \text{clip}\big(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon\big) \tilde{A}_i \right) - \beta D_{\text{KL}}^{(i,t)} \right\} \right]$$
该目标由策略优化项（由组相对优势 $\tilde{A}_i$ 驱动）与正则项（KL 散度控制偏离）组成。

#### 构造 KL 散度估计器
计算 KL 散度项时，需要把它写成单样本可算的形式。  
前向 KL 散度的期望写法为 $\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) = \mathbb{E}_{\pi_\theta}[-\ln u]$，其中 $u = \frac{\pi_{\text{ref}}(y_{i,t})}{\pi_\theta(y_{i,t})} > 0$ 是逐 token 概率比。若直接用单样本 $-\ln u$ 做蒙特卡洛估计，其值可能为负，正则惩罚方向不稳定。

为使逐样本惩罚方向稳定且期望无偏，由引理 1 的切线放缩，构造逐样本非负估计器：
$$D_{\text{KL}}^{(i,t)} = u - \ln u - 1 \ge 0 \quad (\text{当且仅当 } u = 1 \iff \pi_\theta = \pi_{\text{ref}} \text{ 时取等})$$

再验证其在期望意义下无偏：
$$\mathbb{E}_{\pi_\theta}[D_{\text{KL}}^{(i,t)}] = \mathbb{E}_{\pi_\theta}[u] - \mathbb{E}_{\pi_\theta}[\ln u] - 1$$
$$= \sum_a \pi_\theta(a) \frac{\pi_{\text{ref}}(a)}{\pi_\theta(a)} - \mathbb{E}_{\pi_\theta}\left[\ln \frac{\pi_{\text{ref}}}{\pi_\theta}\right] - 1$$
$$= \sum_a \pi_{\text{ref}}(a) - \mathbb{E}_{\pi_\theta}\left[\ln \frac{\pi_{\text{ref}}}{\pi_\theta}\right] - 1$$
$$= 1 + \mathbb{E}_{\pi_\theta}\left[-\ln \frac{\pi_{\text{ref}}}{\pi_\theta}\right] - 1$$
$$= \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$$

这正是 Schulman 给出的 $k_3$ 估计 `[Schulman, 2020]`。同文还有另外两种常用蒙特卡洛估计：
$$k_1 = \ln \frac{\pi_\theta}{\pi_{\text{ref}}}, \qquad k_2 = \frac{1}{2}\left(\ln \frac{\pi_\theta}{\pi_{\text{ref}}}\right)^2, \qquad k_3 = \frac{\pi_{\text{ref}}}{\pi_\theta} - \ln \frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$
其中 $k_1$ 无偏但可取负；$k_2$ 恒非负、方差较低但有偏；$k_3$ 同时无偏且逐样本非负，因而成为路线 B 中的默认选择。

至此，针对 2.3 的核心问题——如何高效、低偏差地估计期望回报 $b^*(x)$——本节给出更直观的统计解法：在局部分布下做组采样，用经验均值估计样本基线，为长程推理场景下的强化学习提供一条更省算力、更稳健的路径。

---

## 3. 跳出参数化

### 3.1 全局目标与逐点解耦

**But, wait, wait, wait. Let us check again** 
让我们重新审视 2.1 一开始给出的目标函数：
$$\max_{\theta \in \mathbb{R}^d} \mathcal{J}(\theta; x) \triangleq \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) r(x, y) - \beta \, \mathbb{D}_{\text{KL}}\big(\pi_\theta(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x)\big)$$

公式中存在一个明显的限制条件 “$\theta$”：从一开始，我们就下意识地假定策略是一个由连续参数 $\theta \in \mathbb{R}^d$ 控制的神经网络 $\pi_\theta$。 
**然而，如果移除这一限制呢？**

$$\max_{\pi \in \Pi} \mathcal{J}_{\text{global}}(\pi) \triangleq \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{y \in \mathcal{Y}} \pi(y \mid x) r(x, y) - \beta \, \mathbb{D}_{\text{KL}}\big(\pi(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x)\big) \right]$$

其中相关空间与函数的形式化定义如下：
* **输入空间**：提示词空间 $\mathcal{X}$，其上附带先验概率测度 $x \sim \mathcal{D}$；
* **输出空间**：离散候选序列空间 $\mathcal{Y}$，其上的概率单纯形记为 $\Delta(\mathcal{Y})$；
* **条件策略全体**：
  $$\Pi \triangleq \big\{ \pi : \mathcal{X} \to \Delta(\mathcal{Y}) \big\} = \prod_{x \in \mathcal{X}} \Delta(\mathcal{Y}) \quad (\Pi \text{ 为所有条件分布构成的笛卡尔积})$$
* **参考策略**：$\pi_{\text{ref}} \in \Pi$，具备全支撑（$\forall x \in \mathcal{X}, y \in \mathcal{Y}, \ \pi_{\text{ref}}(y \mid x) > 0$）。

$\because$ 非参数化后，策略空间 $\Pi$ 具备笛卡尔直积结构，各提示词 $x$ 对应的条件分布 $\pi(\cdot \mid x)$ 彼此独立、无交叉约束，因此最大化全局期望等价于：对每一个 $x$，独立求解 $\mathcal{J}(\pi; x)$ 的极大值：
$$\pi^*(\cdot \mid x) = \arg\max_{\pi(\cdot \mid x) \in \Delta(\mathcal{Y})} \mathcal{J}(\pi; x)$$
其中条件目标为：
$$\mathcal{J}(\pi; x) \triangleq \sum_{y \in \mathcal{Y}} \pi(y \mid x) r(x, y) - \beta \sum_{y \in \mathcal{Y}} \pi(y \mid x) \ln \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \quad \text{s.t.} \ \sum_{y \in \mathcal{Y}} \pi(y \mid x) = 1, \ \pi(y \mid x) \ge 0$$

问题便化为固定上下文 $x$ 下的代数极值问题。这意味着，可以通过代数变换直接求出最优策略 $\pi^*$ 的解析解。

---

### 3.2 难以直接使用的闭式解
由引理 2，当且仅当：
$$\pi(y \mid x) = \pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right) \text{ 时}$$
泛函 $\mathcal{J}(\pi)$ 取得极大值 $\beta \ln Z(x)$。

**然而，虽然我们求出了最优策略的解析形式，却无法直接使用：**
1. 奖励函数 $r(x, y)$ 缺乏解析表达；
2. $Z(x)$ 无法精确求解：输出空间 $\mathcal{Y}$ 的势为 $\mathcal{O}(|\mathcal{V}|^L)$（$|\mathcal{V}|$ 为词表大小，$L$ 为序列长度），精确归一化在实践中不可行。

事实上，第 2 节的种种做法，正是在近似这一理论解析解。 
既然 $r(x, y)$、$Z(x)$ 均无法精确表示或计算，这条路就走不通了吗？

让我们回到最初的问题与假设： 
已知人类相对偏好数据 $\mathcal{D}_{\text{pref}} = \{ (x, y_w, y_l) \}$。 
假定人类在给定上下文 $x$ 下对两个候选 $(y_w, y_l)$ 的选择服从 Bradley–Terry 模型：
$$p^*(y_w \succ y_l \mid x) = \sigma\big(r^*(x, y_w) - r^*(x, y_l)\big), \quad \sigma(z) \triangleq \frac{1}{1 + e^{-z}}$$
又已知：
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

---

## 4. 隐式奖励与极大似然

### 4.1 从点差法说起
于是最核心的问题是：  
已知参考策略 $\pi_{\text{ref}}(y \mid x)$ 与偏好样本 $\mathcal{D}_{\text{pref}}$（其中蕴含对真实奖励 $r^*$ 的偏好信息），分别有偏好模型与最优策略：
$$p^*(y_w \succ y_l \mid x) = \sigma\big(r^*(x, y_w) - r^*(x, y_l)\big) \qquad \text{(Pref)}$$
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right) \qquad \text{(Opt)}$$
求仅关于 $\pi^*$ 的方程。

**相对关系？两式作差？——这让人想起高中圆锥曲线里的点差法：** 
已知方程 $\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1$，两点 $A(x_1, y_1), B(x_2, y_2)$ 均在椭圆上，代入方程：
$$\begin{cases} \frac{x_1^2}{a^2} + \frac{y_1^2}{b^2} = 1 \\ \frac{x_2^2}{a^2} + \frac{y_2^2}{b^2} = 1 \end{cases}$$
两式相减消去右边常数：
$$\frac{(x_1 - x_2)(x_1 + x_2)}{a^2} + \frac{(y_1 - y_2)(y_1 + y_2)}{b^2} = 0$$
进一步得到 $k = \frac{y_1 - y_2}{x_1 - x_2} = -\frac{b^2}{a^2} \cdot \frac{x_1 + x_2}{y_1 + y_2}$，记中点坐标为 $(x_0, y_0)$，得 $k_{AB} \cdot \frac{y_0}{x_0} = -\frac{b^2}{a^2}$。

类似地，既然 $Z(x)$ 难以精确计算，且对 $y$ 是常数，**是否可以用同样思路将其消去？**

---

### 4.2 反解隐式奖励并消去差分常数
对 (Opt) 两边同取自然对数：
$$\ln \pi^*(y \mid x) = \ln\left( \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right) \right) = \ln \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \ln Z(x)$$
整理可得：
$$r^*(x, y) = \beta \ln\left( \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) + \beta \ln Z(x)$$

故：
$$r^*(x, y_w) - r^*(x, y_l) = \left[ \beta \ln\left(\frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}\right) + \beta \ln Z(x) \right] - \left[ \beta \ln\left(\frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right) + \beta \ln Z(x) \right]$$
$$= \beta \ln\left(\frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}\right) - \beta \ln\left(\frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

代入 (Pref)：
$$p^*(y_w \succ y_l \mid x) = \sigma\left( \beta \ln\frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \ln\frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)$$

---

### 4.3 极大似然损失及其梯度
用参数化策略 $\pi_\theta$ 去拟合最优策略 $\pi^*$，把 $\pi_\theta$ 代入上述偏好式，得到模型预测的偏好概率：
$$P_\theta(y_w \succ y_l \mid x) \triangleq \sigma\left( \beta \ln\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \ln\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)$$

在偏好数据 $\mathcal{D}_{\text{pref}}$ 上做极大似然，优化 $\theta$：
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) \triangleq -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \big[ \ln P_\theta(y_w \succ y_l \mid x) \big]$$
代入预测偏好概率，得到最终损失：
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \ln \sigma\left( \beta \ln\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \ln\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

再对 $\theta$ 求梯度，以分析该损失函数优化机制。定义模型隐式奖励：
$$\hat{r}_\theta(x, y) \triangleq \beta \ln\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$
利用 $\nabla_z \ln \sigma(z) = 1 - \sigma(z) = \sigma(-z)$ 做链式求导：
$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \sigma\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big) \cdot \nabla_\theta\left( \beta \ln \pi_\theta(y_w \mid x) - \beta \ln \pi_\theta(y_l \mid x) \right) \right]$$
$$= -\beta \, \mathbb{E}_{(x, y_w, y_l)} \left[ \underbrace{\sigma\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big)}_{\text{自适应权重 } w} \cdot \left( \underbrace{\nabla_\theta \ln \pi_\theta(y_w \mid x)}_{\text{提高胜出项概率}} - \underbrace{\nabla_\theta \ln \pi_\theta(y_l \mid x)}_{\text{压低失败项概率}} \right) \right]$$

**该梯度有两个直观性质：**
1. **更新方向明确**：沿拉高胜出项 $y_w$ 概率、压低落选项 $y_l$ 概率的方向更新；
2. **自适应加权**：前置权重 $w = \sigma(\hat{r}_l - \hat{r}_w)$ 度量当前预测偏差：
 * 若模型当前完全判断错误 ($\hat{r}_l \gg \hat{r}_w$)，则 $w \to 1$，梯度较大、纠正力度大；
 * 若模型已能很好区分偏好 ($\hat{r}_w \gg \hat{r}_l$)，则 $w \to 0$，梯度自然衰减，减轻过拟合。

---

## 5. 小结

至此，通过反解最优策略的解析形式，并借助 Bradley–Terry 偏好假设，我们得到一种与第 2 节“显式奖励 + 在线强化学习”在目标上数学等价的直接优化途径——前提是偏好服从该模型、真实奖励与最优策略一致，且在线目标下的最优策略与人类标签同向。借助“隐式奖励 + 离线监督”，便能以简洁直接的方式，用人类相对偏好数据训练模型生成更受偏好的回答，从而回答文首问题。

**然而必须指出：第 3–4 节的方法建立在一系列理想化假设上：**
1. **人类偏好服从 Bradley–Terry 模型**，即相对优劣由奖励差决定，且偏好具有传递性（真实数据却常有噪声与互相矛盾）；
2. **可实现性**：真实奖励落在奖励模型类内，最优策略 $\pi^*$ 也能被策略模型表示（否则会出现设定失配）；
3. **最优策略与人类偏好同向**：第 4 节方法与第 2 节在线目标的等价性是有条件的；若不满足该隐含条件，优化目标便不再等价。

正因如此，第 2 节的在线强化学习路径在长推理训练中仍不可替代：它仍是促使模型自发回溯、反思与自我纠错，并有效利用测试时计算的重要手段。R1-Zero 等结果也提示：在可验证奖励设定下，思维链是否好读，输出是否格式良好、语言统一，未必与任务表现同向。

---

## 对号入座

最后，揭晓前文所得结果的“真名”：

* **建模假设 · Bradley–Terry 偏好模型**：`[Bradley and Terry, 1952]`
* **$\S 1$ 策略梯度与 REINFORCE**：`[Williams, 1992]`；**策略梯度定理**：`[Sutton et al., 1999]`
* **$\S 1.1$ DeepSeek-R1-Zero（零监督微调、可验证奖励上的大规模强化学习）**：`[DeepSeek-AI, 2025]`
* **$\S 2.4$ 路线 A → PPO / RLHF**：
 * **偏好强化学习**：`[Christiano et al., 2017]`
 * **广义优势估计 (GAE)**：`[Schulman et al., 2016]`
 * **近端策略优化 (PPO)**：`[Schulman et al., 2017]`
 * **InstructGPT**：`[Ouyang et al., 2022]`
* **$\S 2.5$ 路线 B → GRPO**：`[Shao et al., 2024]`
* **$\S 2.5$ 逐 token KL 散度估计 $k_1,k_2,k_3$**：`[Schulman, 2020]`
* **$\S 4$ 隐式奖励与极大似然 → DPO**：`[Rafailov et al., 2023]`

**感谢前辈们的探索与开拓。**

---

\newpage

## 参考文献

1. **Bradley, R. A., & Terry, M. E. (1952).** Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika, 39*(3/4), 324–345. https://doi.org/10.1093/biomet/39.3-4.324

2. **Williams, R. J. (1992).** Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning, 8*(3–4), 229–256. https://doi.org/10.1007/BF00992696

3. **Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (1999).** Policy gradient methods for reinforcement learning with function approximation. In S. A. Solla, T. K. Leen, & K. Müller (Eds.), *Advances in Neural Information Processing Systems 12* (pp. 1057–1063). MIT Press. https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html

4. **Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2016).** High-dimensional continuous control using generalized advantage estimation. In *International Conference on Learning Representations (ICLR)*. arXiv:1506.02438. https://arxiv.org/abs/1506.02438

5. **Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017).** Deep reinforcement learning from human preferences. In I. Guyon et al. (Eds.), *Advances in Neural Information Processing Systems 30*. Curran Associates. arXiv:1706.03741. https://arxiv.org/abs/1706.03741

6. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** Proximal policy optimization algorithms. *arXiv preprint* arXiv:1707.06347. https://arxiv.org/abs/1707.06347

7. **Schulman, J. (2020, March 7).** Approximating KL divergence. http://joschu.net/blog/kl-approx.html

8. **Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022).** Training language models to follow instructions with human feedback. In S. Koyejo et al. (Eds.), *Advances in Neural Information Processing Systems 35*. Curran Associates. arXiv:2203.02155. https://arxiv.org/abs/2203.02155

9. **Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023).** Direct preference optimization: Your language model is secretly a reward model. In A. Oh et al. (Eds.), *Advances in Neural Information Processing Systems 36*. Curran Associates. arXiv:2305.18290. https://arxiv.org/abs/2305.18290

10. **Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024).** DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint* arXiv:2402.03300. https://arxiv.org/abs/2402.03300

11. **DeepSeek-AI. (2025).** DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. *arXiv preprint* arXiv:2501.12948. https://arxiv.org/abs/2501.12948  
    另见正式出版版本：DeepSeek-AI et al. (2025). DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. *Nature*. https://doi.org/10.1038/s41586-025-09422-z
