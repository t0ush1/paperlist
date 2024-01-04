- 网约车即时激励优化（Instant Incentive Optimization，IIO）问题
- 问题过程
  - 第一步：价格计算，根据行程距离和预计时间等因素，确定乘客的票价以及司机的赔偿
  - 第二步：即时激励优化，给乘客折扣，给司机红利
- 建模为 0-1 整数规划问题，利用拉格朗日对偶求解
- 根据历史数据调整参数（拉格朗日乘子），在每个请求出现时给出一个在线决策



## Formulation

- 乘客
  - 请求：$n\in N$
  - 车费：$F_n$
  - 激励（discount）
    - $d_1<d_2<\dots <d_I$ 共 $I$ 个等级
    - $d_1=0$，表示没有激励
  - 付款：$F_n(1-d_i)$，其中 $d_i$ 是选择的激励
- 司机
  - 补偿：$C_n$
  - 激励（bonus）
    - $b_1<b_2<\dots<b_J$ 共 $J$ 个等级
    - $b_1=0$，表示没有激励
  - 收益：$C_n(1+b_j)$，其中 $b_j$ 是选择的激励
- 平台
  - 目标：最大化完成的请求数量
  - 约束
    - 总激励不超过共享预算 $B$
    - 对于每个请求，司机的收益不能高于乘客的付款



## The Prediction Model

- $p_{nij}$，乘客的激励等级为 $i$，司机的激励等级为 $j$ 时，请求 $n$ 完成的概率

- $f_n$，其他影响概率的特征

- 预测模型
  $$
  \hat{p}_{nij}=\hat{h}(d_i,b_j,f_n)
  $$
  其中 $\hat{p}_{nij}$ 是 $p_{nij}$ 的估计，$\hat{h}(*)$ 取决于特定的模型结构（无需关心）



## Binary Integer Programming Formulation

- 整数规划表示为
  $$
  \begin{aligned}
  \max\limits_x &\sum_{n\in N}\sum_{i=1}^I\sum_{j=1}^Jx_{nij}\hat{p}_{nij},\\
  \text{s.t.} &\sum_{n\in N}\sum_{i=1}^I\sum_{j=1}^Jx_{nij}\hat{p}_{nij}(F_nd_i+C_nb_j)\le B,\\
  &\sum_{i=1}^I\sum_{j=1}^Jx_{nij}\left[C_n(1+b_j)-F_n(1-d_i)\right]\le 0,\forall n,\\
  &\sum_{i=1}^I\sum_{j=1}^Jx_{nij}=0,\forall n,\\
  &x_{nij}\in\{0,1\},\forall n,i,j
  \end{aligned}
  $$
  其中 $x=(x_{111},\dots,x_{1IJ},\dots)^T$ 为决策变量，$x_{nij}=1$ 表示第 $n$ 个请求的乘客和司机的激励等级分别为 $i$ 和 $j$

- 对于 $n\in N$，假设 $C_n<F_n$，则至少有一个可行解 $x_{n11}=1,x_{nij}=0,i,j\neq 1$