- 交叉动态任务分配（cross dynamic task assignment，CDTA），空间众包
- 场景：跨平台空间众包中，任务可以分配给内部员工或合作平台的外部员工
- 贡献
  - 引入员工的声誉分数，形式化定义 CDTA
  - 框架：基于跨平台的激励机制和混合批处理策略
  - 算法：...，最大化平台内部效用
  - 博弈相关
    - 建模 CDTA 为位势博弈（Potential Game），证明其具有纯策略纳什均衡
    - 提出一种博弈论算法，最大化外部员工收入



## Problem Statement

- 空间任务：$r=<t_r,l_r,d_r,p_r>\in R$
  - 提交时间：$t_r$，截止时间：$t_r+d_r$
  - 位置：$l_r$
  - 报酬：$p_r$（给平台）
  
- 内部员工：$w_{in}=<t_{w_{in}},l_{w_{in}},d_{w_{in}},rad_{w_{in}},\rho_{w_{in}}>\in W_{in}$
  - 到达时间：$t_{w_{in}}$，等待时间：$d_{w_{in}}$，离开时间：$t_{w_{in}}+d_{w_{in}}$
  - 位置：$l_{w_{in}}$
  - 服务半径：$rad_{w_{in}}$
  - 信誉分：$\rho_{w_{in}}\in[0,1]$，取决于历史完成的任务数量和评分
  
- 外部员工：$w_{out}=<t_{w_{out}},l_{w_{out}},d_{w_{out}},rad_{w_{out}},\rho_{w_{out}}>\in W_{out}$，同上

- 外部报酬： $p'(w_{out},r)\in(0,p_r]$（给外部员工）

- 内部效用

  - 给定任务分配集合 $M$，内部效用 $U_M$ 主要取决于平台收入和员工信誉分

  - 当内部员工 $w_{in}$ 完成任务 $r$ 时，平台收入 $p_r$

    当外部员工 $w_{out}$ 完成任务 $r$ 时，平台收入 $p_r-p'_r$

  - 内部效用定义为
    $$
    U_M=\sum_{(r,w_{in})\in M}(p_r\times\rho_{w_{in}})+\sum_{(r,w_{out})\in M}\left((p_r-p'_r)\times \rho_{w_{out}}\right)
    $$

- CDTA 问题：找到 $M$ 使得 $U_M$ 最大化并满足以下约束

  - 每项任务只能被一个员工完成
  - 任务应在工人离开平台前出现，工人应在任务截止日期前到达平台
  - 任务被分配后不可更改
  - 员工只能完成在服务半径内的任务



## SOLUTION FRAMEWORK

### A Cross-platform Incentive Mechanism

- 员工短缺度：给定地区 $Re$ 和时间戳 $T$，员工短缺度定义为
  $$
  SD_{Re}^T=\left\{\begin{array}{l}
  0, &\text{if } p>k \text{ or } n_r=0\\
  1, &\text{if } n_w=0\\
  -\tanh(\ln(p/k)), &\text{otherwise}
  \end{array}\right.
  $$
  其中 $n_w$ 为员工数，$n_r$ 为任务数，$k$ 为需要的员工-任务比，$p=n_w/n_r$ 为实际的员工-任务比

- 外部报酬：给定外部员工 $w$ 和任务 $r$，外部报酬定义为
  $$
  p'(w,r)=\alpha p_r\rho_w+\beta\sum_{r'\in \R}p_{r'}\frac{SD_r}{\sum_{r'\in\R}SD_{r'}}
  $$
  其中 $\R$ 是所有外部任务集合，$\alpha,\beta\in[0,1]$ 是用来调整权重的参数且 $\alpha+\beta\le 1$（保证总外部报酬小于总任务报酬？）

  - 第一项：基本报酬，取决于任务报酬和员工信誉
  - 第二项：额外红利，在低员工密度区域能够激励更多的外部员工去执行任务，同时保证定价的公平性

- 外部收入：给定外部员工 $w$ 和任务 $r$，其收入定义为
  $$
  Rev(w,r)=p'(w,r)-c_w\times d(w,r)
  $$
  其中 $c_w$ 是移动的单位成本，$d(w,r)$ 是 $w$ 到 $r$ 的欧氏距离



## Game-theoretic Algorithm

目的：在不影响内部效用的情况下，让外部工人收入最大化

纳什均衡：当其他外部员工都停留在分配给他们的任务上时，没有一个外部员工可以通过单方面从分配的任务切换到另一个任务来增加收入



博弈模型

- 考虑 $n$ 个参与者的博弈
  $$
  G=<W_o,\{S_w\}_{w\in W_o},\{U_w:\times_{w\in W_o}S_w\}_{w\in W_0}\to\R>
  $$

- 参与人：$w\in W_o$

- 策略：$s_w\in S_w$

  - $S_w\subseteq R$，$R$ 为任务集合
  - $s_w$ 代表 $w$ 处理的任务

- 收益函数
  $$
  \begin{aligned}
  U_w(s_w,s_{-w})=
  &p'(w,s_w)-c_w\times d(w,s_w)\\
  &=\alpha p_{s_w}\rho_w+\beta\sum_{s_{\tilde{w}}\in \mathbb{S}}p_{s_{\tilde{w}}}\frac{SD_w}{\sum_{s_{\tilde{w}}\in\mathbb{S}}SD_{\tilde{w}}}-c_w\times d(w,s_w)
  \end{aligned}
  $$
  其中 $\mathbb{S}=\{s_w,s_{-w}\}$

- 位势博弈：若存在势函数 $\Phi:\{S_w\}_{w\in W_o}\to\R$ 使得对 $\forall w\in W_o,\forall s_w,s'_w\in S_w,\forall s_{-w}\in S_{-w}$，满足
  $$
  U_w(s_w,s_{-w})-U_w(s'_w,s_{-w})=\Phi(s_w,s_{-w})-\Phi(s'_w,s_{-w})
  $$
  则称其为位势博弈，一定存在纯策略纳什均衡



证明：定义 $\Phi$ 为 ... ，带入即可证明
