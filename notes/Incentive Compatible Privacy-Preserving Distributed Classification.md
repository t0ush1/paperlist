重点：

- 博弈论 + 分布式数据挖掘 -> 真实数据共享
- 非合作博弈：VCG，合作博弈：沙普利值
- 不需要验证原始数据



## 博弈相关定义

- 有限参与人，单回合，同时行动，不完全信息博弈

- 符号

  - $n$ 个参与人
  - outcome（结果）：$a\in A$
  - valuation function（估价函数）：$v_i\in V_i$
    - $v_i$ 表示原始估价（类型），$v_i(a)$ 表示在结果 $a$ 下的原始估价（逆映射+投影？）
    - $V=V_1\times V_2\times\dots\times V_n$
  - mechanism（社会选择函数）：$f:V\rightarrow A$
  - payment function（支付函数）：$p_i:V\to\Re$
  - 效用函数：$v_i(a)-p_i(v_i,v_{-i})$

- 激励相容：$\forall i=1,\dots,n$，$\forall v\in V$，$\forall v'_i\in V_i$，令 $a=f(v_i,v_{-i})$，$a'=f(v'_i,v_{-i})$，则需满足
  $$
  v_i(a)-p_i(v_i,v_{-i})\ge v_i(a')-p_i(v'_i,v_{-i})
  $$

- 个人理性：参与博弈比不参与（效用为 0）要好

- VCG 机制：满足

  - 配置效率：$f(v)=\arg\max_{a\in A}\sum v_i(a)$
  - 格罗夫斯支付规则：$p_i(v)=h(v_{-i})-\sum_{j\neq i}v_j(f(v))$，其中 $h_i:V_{-i}\to\Re$



## 论文的模型：Assured Information Sharing Game

Mediated Information Sharing Game

- 参与方：$P_1,P_2,\dots,P_n$，中介 $P_t$
- 前提
  - $x_i$ 为 $P_i$ 用于共享的一组原始数据
  - $P_t$ 用安全的方式从参与方提供的数据中计算模型，并且有一个独立的测试集
- 过程
  - 每个参与方选择数据 $x'_i$ ；$X$ 为原始数据向量，$X'$ 为选择数据向量
  - $X'$ 被提交到 $P_t$ ，通过函数 $D$ 来进行安全计算
  - 每个参与方得到计算结果 $m=D(X')$
- 支付
  - 效用函数 $u_i(x_i,m)=\max\{v_i(m)-v_i(D(x_i)), 0\}-p_i(X',m)-c(D)$
  - $v_i(m)=acc(m)$ ，为数据挖掘模型的准确度（越大越好）
  - $v_i(D(x_i))$ 为只基于原始数据的模型的精度
  - $0$ 为保留效用，即不参与博弈的收益
  - （没看懂这里怎么设计的）
  - $p_i(X',m)$ 为 $P_i$ 支付的钱，小于 0 表示得到钱
  - $c(D)$ 表示函数计算的开销



## 结论及证明

- 前提

  - $P_t$ 还计算了所有 $D(X'_{-i})$
  - 令 $p_i(X',m)=\sum_{j\neq i}v_j(D(X'_{-i}))-\sum_{j\neq i}v_j(m)-c(D)$
  - $v_i$ 是根据 $P_t$ 拥有的独立测试集测出的
  - $P_i$ 得到了一笔钱，等于边际贡献
  - $-c(D)$ 是为了抵消计算开销

- 假设：$x_i$ 和 $x'_i$ 的差异越大，精度增加的可能性越小（？）

  即对于 $X=x_i\cup X_{-i},X'=x'_{-i}\cup X_{-i}$，有
  $$
  E\left[acc(D(X))\right]\ge E\left[acc(D(X'))\right]+f(dist(X,X'))
  $$
  其中 $E$ 为期望值，$f$ 为非负递增函数

- 计算得
  $$
  u_i(x_i,D(X))=\max\{v_i(D(X))-v_i(D(x_i)), 0\}-\sum_{j\neq i}v_j(D(X_{-i}))+\sum_{j\neq i}v_j(D(X))\\
  u_i(x_i,D(X'))=\max\{v_i(D(X'))-v_i(D(x_i)), 0\}-\sum_{j\neq i}v_j(D(X_{-i}))+\sum_{j\neq i}v_j(D(X'))
  $$

- 激励相容：证明 $E\left[u_i(x_i,D(X))\right]\ge E\left[u_i(x_i,D(X'))\right]$，即
  $$
  E\left[\max\{v_i(D(X))-v_i(D(x_i)), 0\}\right]+E\left[\sum_{j\neq i}v_j(D(X))\right]\ge\\
  E\left[\max\{v_i(D(X'))-v_i(D(x_i)), 0\}\right]+E\left[\sum_{j\neq i}v_j(D(X'))\right]
  $$
  由 $E\left[v_k(D(X))\right]\ge E\left[v_k(D(X'))\right],\forall k$，结论成立

- 个人理性：证明 $E\left[u_i(x_i,D(X))\right]\ge 0$，即
  $$
  E\left[\sum_{j\neq i}v_j(D(X))-\sum_{j\neq i}v_j(D(X_{-i}))\right]\ge 0
  $$
  由 $E\left[v_k(D(X))\right]-E\left[v_k(D(X'))\right]=f(dist(X,X'))\ge 0,\forall k$，结论成立