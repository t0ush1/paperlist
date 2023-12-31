重点：

本地模型的上传成本太高，提出两种改进方法

- 结构化更新（structured update）：从有限空间中学习更新，参数更少
- 草图化更新（sketched update）：学习完整模型更新，上传前压缩



## 形式化表述

- 目标：从大量存储在客户端的数据中学习出模型

  - 模型参数记为矩阵 $W\in\R^{d_1\times d_2}$
  - $d_1$ 和 $d_2$ 分别为输出和输入维度

- 过程：在第 $t\ge 0$ 轮中，

  - 服务器分发当前模型 $W_t$ 给包含 $n_t$ 个客户端的一个子集 $S_t$

  - 客户端独立地根据本地数据更新模型

    - 更新后的模型记为 $W_t^1,W_t^2,\dots,W_t^{n_t}$
    - 更新记为 $H_t^i:=W_t^i-W_t$，$\forall i\in S_t$

  - $S_t$ 将更新发回给服务器，服务器聚合所有本地更新来计算全局更新
    $$
    W_{t+1}=W_t+\eta_tH_t,\quad H_t:=\frac1{n_t}\sum_{i\in S_t}H_t^i
    $$
    其中 $\eta_t$ 为学习率



## Structured Update

### Low rank

- 将 $H_t^i\in\R^{d_1\times d_2}$ 降秩，使其不超过 $k$
  - 令 $H_t^i=A_t^iB_t^i$，其中 $A_t^i\in\R^{d_1\times k}$，$B_t^i\in \R^{k\times d_2}$
  - $A_t^i$ 随机生成的常量，只优化 $B_t^i$
- 每轮中为每个客户端独立地生成 $A_t^i$
- 客户端只需发送 $B_t^i$ 到服务器，节省了 $d_1/k$ 的参数

### Random mask

- 将 $H_t^i$ 限定为稀疏矩阵，并遵循预先定义的随机稀疏模式（即随机掩码）
- 每轮中每个客户端独立地生成稀疏模式
- 客户端只需发送 $H_t^i$ 的非 0 元素，以及随机种子



## Sketched Update

### Subsampling

### Probabilistic quantization

### Improving the quantization by structured random rotations