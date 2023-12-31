重点：

- 设计 DFL 场景下的激励函数（收益函数），满足不同方可自定义
- 将其建模成马尔科夫博弈问题，其解是行动的概率分布
- 通过多智能体强化学习（MARL）方法来求纳什均衡解，且使得激励函数最大化



## 前置知识

### 分布式联邦学习

- 多方：$\mathcal{V}$，迭代步骤：$t$，每次迭代有如下三个阶段：

- local training：每方 $i$ 从本地数据集 $\mathcal{D}_i$ 中采样 $\{\xi_m^i\}_{m=1}^M$，其中 $M$ 是批大小，再计算梯度
  $$
  \Delta\theta_i(t)=\sum_{m=1}^M\nabla f_i(\theta_i;\xi_m^i)\quad\forall i\in\mathcal{V}
  $$
  其中 $\theta_i$ 为模型参数，$f_i(\cdot)$ 为损失函数；因此，模型的更新为
  $$
  \theta_i(t+\frac12)=\theta_i(t)-\beta\Delta\theta_i(t)\quad\forall i\in\mathcal{V}
  $$
  其中 $\beta$ 为学习率

- gossiping：每方 $i$ 的邻居为 $\mathcal{N}(i)$，与其子集 $\widehat{\mathcal{N}}(i)\subset\mathcal{N}(i)$ 交流来拉取他们的模型参数 $\theta_j(t+\frac12),\forall j\in\widehat{\mathcal{N}}(i)$

- averaging：每方 $i$ 根据本地模型和拉取的模型的加权平均来更新参数
  $$
  \theta_{i}(t+1)=(1-\sum_{j\in\widehat{\mathcal{N}}(i)}\rho_{i})\theta_{i}(t+\frac{1}{2})+\sum_{j\in\widehat{\mathcal{N}}(i)}\rho_{i}\theta_{j}(t+\frac{1}{2})\quad\forall i\in\mathcal{V}
  $$
  其中 $\rho_i$ 是整合邻居模型的平均率

### 多智能体强化学习

- 马尔科夫博弈（随机博弈）：$\{\mathcal{V},\mathcal{S},\{\mathcal{A}_i\}_{i\in\mathcal{V}},\mathcal{P},\{R_i\}_{i\in\mathcal{V}},\gamma\}$

  - $\mathcal{V}=\{1,\dots,|\mathcal{V}|\}$：智能体集合
  - $\mathcal{S}$，状态空间
  - $\mathcal{A}_i$，智能体 $i$ 的行动空间，$\mathcal{A}:=\mathcal{A}_1\times\cdots\times\mathcal{A}_{|\mathcal{V}|}$
  - $\mathcal{P}:\mathcal{S}\times\mathcal{A}\to\mathcal{S}$，转移概率函数
  - $R_i:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to\R$，智能体 $i$ 采取某一行动时的即时奖励
  - $\gamma$，折扣因子

- 纳什均衡策略：$\pi^{*}=(\pi_{1}^{*},\cdots,\pi_{|\mathcal{V}|}^{*})$

  - $\pi_i:\mathcal{S}\to\mathcal{A}_i$

  - 能最大化个体长期奖励
    $$
    \mathbb{E}\left[\sum_{t=0}^\infty\gamma^tR_i(s_t,a_t,s_{t+1})\mid a_t\sim\pi^*\right]
    $$
    其中 $s_t,a_t,s_{t+1}$ 分别表示当前状态、被执行的行动、下一个状态



## IDEA 框架

### 合作方

将 DFL 中的合作方用无向图表示：$\mathcal{G}=\{\mathcal{V},\mathcal{E}\}$

- $\mathcal{V}$ ，点集；$\mathcal{E}$ ，边集
- $i\in\mathcal{V}$ ，第 $i$ 方
- $\mathcal{N}(i)$，邻居
- $\mathcal{D}_i$，数据集
- $p_i^{cpt}$，计算能力；$p_i^{cmt}$，交流能力
  - 与数据处理速率和数据传输速率有关
- 神经网络模型参数 $\theta_i$
  - 每方的 NN 模型是相同的

### 自适应合作策略

合作策略用 $\pi_i$ 表示，输出行动 $a_i(t)$ 的概率分布，且
$$
a_i(t)\sim\pi_i=[a_{i,0}(t),a_{i,1}(t),\cdots,a_{i,|\mathcal{N}(i)|}(t)]
$$
其中 

- $a_{i,0}(t)=1$ 表示 $i$ 在 $t$ 时刻训练了本地模型，否则 $a_{i,0}(t)=0$
- $a_{i,j}(t)=1$ 表示 $i$ 在 $t$ 时刻拉取了邻居 $j\in\mathcal{N}(i)$ 的模型参数，否则 $a_{i,j}=0$

### 可定制奖励方案

- 本地训练成本
  $$
  c_i^l(t)=a_{i,0}(t)\cdot c_i^{train}\cdot |\mathcal{D}_i|/p_i^{cpt}\quad\forall i\in\mathcal{V}
  $$
  其中 $c_i^{train}$ 是本地模型训练的单位成本

- 拉取参数成本
  $$
  c_i^p(t)=\sum_{j\in\mathcal{N}(i)}a_{i,j}(t)pr_j(t)\quad\forall i\in\mathcal{V}
  $$
  其中 $Pr=[pr_{i}]_{i\in\mathcal{V}}\in\R^{|\mathcal{V}|}$ 为各方的参数价格向量，$pr_i$ 表示 $i$ 的参数价格，且
  $$
  pr_i(t)=c_i^{com}\frac{S}{p_i^{cmt}}+c_i^{acc}\log(1+acc_i(t))\quad\forall i\in\mathcal{V}
  $$
  其中 $S$ 是模型参数大小，$c_i^{com}$ 是单位交流成本，$c_i^{acc}$ 是精度 $acc_i(t)$ 的优先级度量

- 推送参数收益
  $$
  r_i^p(t)=\sum_{j\in\mathcal{N}(i)}a_{j,i}(t)pr_i(t)\quad\forall i\in\mathcal{V}
  $$
  与拉取参数成本相对应

- 服务收益
  $$
  r_i^s(t)=\log(1+acc_i(t))\quad\forall i\in\mathcal{V}
  $$
  随 $acc_i(t)$ 单增且是严格凹函数

- 激励函数
  $$
  R_i(t)=\beta_sr_i^s(t)+\beta_pr_i^p(t)-\beta_cc_i^p(t)-\beta_lc_i^l(t)\quad\forall i\in\mathcal{V}
  $$
  其中 $\beta_s$，$\beta_p$，$\beta_c$，$\beta_l$ 分别表示各项权重，可通过调整权重来定制激励方案



## 激励模型设计

### 激励机制

- 马尔科夫博弈建模：$\{\mathcal{V},\mathcal{S},\{\mathcal{A}_i\}_{i\in\mathcal{V}},\mathcal{P},\{R_i\}_{i\in\mathcal{V}},\gamma\}$

  - 状态空间 $\mathcal{S}$ 为所有方模型精度的组合，令
    $$
    s(t)=[acc_{i,c}(t-1)]_{i\in\mathcal{V},c\in C}
    $$
    其中 $acc_{i,c}(t-1)$ 是 $i$ 在 $t-1$ 时刻在类别 $c$ 上的精度，$C$ 是数据集的类集

  - 行动 $a_i(t)\in\mathcal{A}_i$ 的定义参上

  - 转移概率 $p$ 满足
    $$
    \sum_{s(t+1)\in\mathcal{S}}p\left(s(t+1)|s(t),a_1(t),\dots,a_{|\mathcal{V}|}(t)\right)=1
    $$

  - 即时奖励 $R_i(t)$ 直接定义为激励函数，参上

- 目标函数（？？？）
  $$
  \max U^i(\pi_i|\pi_{-i})=\mathbb{E}_{\pi_i|\pi_{-i}}\left[\sum_{t=0}^\infty \gamma_tR_i(t)\right]
  $$

- 效用函数（value-function）：$V^i:\mathcal{S}\to\R$（公式里点竖st是啥意思？？？）
  $$
  V_{\pi_i,\pi_{-i}}^i(\mathbf{s}):=\mathbb{E}\left[\sum_{t\ge 0}\gamma^tR_i(s(t),a(t),s(t+1))|a_i(t)\sim\pi_i(\cdot|s(t)),s(0)=s\right]
  $$

- 行动效用函数（q-function）：$Q^i:\mathcal{S}\times\mathcal{A}_i\to\R$
  $$
  Q_{\pi_i,\pi_{-i}}^i(s,a):=\mathbb{E}\left[\sum_{t\ge 0}\gamma^tR_i(s(t),a(t),s(t+1))|a_i(t)\sim\pi_i(\cdot|s(t)),s(0)=s,a(0)=a\right]
  $$

- 纳什均衡：$\pi^{*}=(\pi_{1}^{*},\cdots,\pi_{|\mathcal{V}|}^{*})$ 满足 $\forall s\in\mathcal{S}$，$\forall i\in\mathcal{V}$，$\forall\pi_i$，有
  $$
  V_{\pi_{i}^{*},\pi_{-i}^{*}}^{i}(s)\geq V_{\pi_{i},\pi_{-i}^{*}}^{i}(s)
  $$

- 对于有限空间、无限步的折扣马尔科夫博弈，纳什均衡一定存在

### 合作策略学习

基于MARL，提出一种近似方法，简化 Q 函数的计算