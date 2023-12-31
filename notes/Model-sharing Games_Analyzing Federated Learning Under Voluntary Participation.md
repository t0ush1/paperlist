## 线性回归模型

| 符号                       | 类型            | 解释   |
| -------------------------- | --------------- | ------ |
| $\theta_j$，$\hat\theta_j$ | $D\times 1$     | 列向量 |
| $\epsilon_j^2$             | $1\times 1$     | 标量   |
| $X_j$                      | $n_j\times D$   | 矩阵   |
| $x$                        | $1\times D$     | 行向量 |
| $\Sigma_j$，$X^T_jX_j$     | $D\times D$     | 方阵   |
| $Y_j$，$\eta_j$            | $n_j\times 1$   | 列向量 |
| $\eta_j\eta_j^T$           | $n_j\times n_j$ | 方阵   |



- 参与人：$j\in[M]$

- 每个参与人提取参数，独立同分布
  - $(\theta_j,\epsilon_j^2)\sim \Theta$
  - $\theta_j$ 为 $D$ 个参数
    - 第 $d$ 个分量为 $\theta_j^d$
    - 记 $\sigma_d^2=Var(\theta_j^d)$
  - $\epsilon_j^2$ 为采样时的噪声，记 $\mu_e=\mathbb{E}_{(\theta_j,\epsilon_j^2)\sim \Theta}[\epsilon_i^2]$

- 每个参与人从输入分布中采样，数据点数为 $n_j$

  - 输入：$X_j\sim\mathcal{X}_j$
  - 测试点：$x\sim\mathcal{X}_j$，记 $\Sigma_j=\mathbb{E}_{x\sim\mathcal{X}_j}[x^Tx]$
  - 输出：$Y_j\sim\mathcal{D}_j(X_j\theta_j,\epsilon_j^2)$
    - $\mathcal{D}_j$ 为期望为 $X_j\theta_j$，方差为 $\epsilon_j^2$ 的某种分布
  - 残差：$\eta_j$，满足 $Y_j=X_j\theta_j+\eta_j$

- 使用 OLS（最小二乘法）估计参数
  $$
  \hat{\theta_j}=(X_j^TX_j)^{-1}X_j^TY_j
  $$
  要求 $X_j^TX_j$ 为可逆矩阵，即 $\Sigma_j$ 可逆的概率为 1

- 单点误差
  $$
  (x\hat\theta_j-x\theta_j)^2
  $$

- 均值估计：$D=1$，$\mathcal{X}_j$ 为始终 $1$ 的分布



## 本地估计的 MSE 的期望

MSE：均方误差

### 结论

对于线性回归，$n_j$ 个样本的本地估计的 MSE 的期望 $\mathbb{E}[(x\hat\theta_j-x\theta_j)^2]$ 为
$$
\mu_e\cdot\tr\left[\Sigma_j\mathbb{E}_{X_j\sim\mathcal{X}_j}\left[\left(X_j^TX_j\right)^{-1}\right]\right]
$$

如果 $\mathcal{X}_j$ 为均值为 0 的正态分布，则可简化为
$$
\frac{\mu_e}{n_j-D-1}D
$$
如果是对于均值估计，则可简化为
$$
\frac{\mu_e}{n_j}
$$

### 证明1

先计算
$$
\begin{aligned}
x\hat\theta_j-x\theta_j
&=x\left(\left(X_j^TX_j\right)^{-1}X_j^TY_j-\theta_j\right)\\
&=x\left(\left(X_j^TX_j\right)^{-1}X_j^T\left(X_j\theta_j+\eta_j\right)-\theta_j\right)\\
&=x\left(\theta_j+\left(X_j^TX_j\right)^{-1}X_j^T\eta_j-\theta_j\right)\\
&=x\left(X_j^TX_j\right)^{-1}X_j^T\eta_j
\end{aligned}
$$

再考虑矩阵的迹的性质

- 对任意标量，$a=\tr(a)$
- $\tr(AB)=\tr(BA)$
- $\mathbb{E}[\tr(A)]=\tr(\mathbb{E}[A])$

因此
$$
\begin{aligned}
\left(x\hat\theta_j-x\theta_j\right)^2
&=\tr\left[\left(x\hat\theta_j-x\theta_j\right)^2\right]\\
&=\tr\left[\left(x\hat\theta_j-x\theta_j\right)^T\left(x\hat\theta_j-x\theta_j\right)\right]\\
&=\tr\left[\left(x\hat\theta_j-x\theta_j\right)\left(x\hat\theta_j-x\theta_j\right)^T\right]\\
&=\tr\left[x\left(X_j^TX_j\right)^{-1}X_j^T\eta_j\eta_j^TX_j\left(X_j^TX_j\right)^{-1}x^T\right]
\end{aligned}
$$

通过计算期望来化简。考虑 $n_j\sim\mathcal{D}_j(0,\epsilon_j^2)$，得上式
$$
=\tr\left[x\left(X_j^TX_j\right)^{-1}X_j^TVX_j\left(X_j^TX_j\right)^{-1}x^T\right]
$$
其中 $V=\mathbb{E}_{\eta_j\sim\mathcal{D}_j(0,\epsilon_j^2)}[\eta_j\eta_j^T]$，矩阵 $\eta_j\eta_j^T$ 有以下性质

- $n_j\times n_j$ 方阵
- 主对角线元素为 $(\eta_j^k)^2$，故 $\mathbb{E}[(\eta_j^k)^2]=\mathbb{E}[\eta_j^k]^2+Var(\eta_j^k)=\epsilon_j^2$
- 其他元素为 $\eta_j^k\cdot\eta_j^l$，$k\neq l$，故 $\mathbb{E}[\eta_j^k\cdot\eta_j^l]=\mathbb{E}[\eta_j^k]\cdot\mathbb{E}[\eta_j^l]=0$（相互独立）

因此 $V=diag(\epsilon_j^2,\epsilon_j^2,\dots,\epsilon_j^2)$，故上式
$$
\begin{aligned}
&=\epsilon_j^2\tr\left[x\left(X_j^TX_j\right)^{-1}X_j^TX_j\left(X_j^TX_j\right)^{-1}x^T\right]\\
&=\epsilon_j^2\tr\left[x\left(X_j^TX_j\right)^{-1}x^T\right]\\
&=\epsilon_j^2\tr\left[x^Tx\left(X_j^TX_j\right)^{-1}\right]
\end{aligned}
$$
对于 $(\theta_j,\epsilon_j^2)\sim \Theta$，有 $\mathbb{E}_{(\theta_j,\epsilon_j^2)\sim\Theta}[\epsilon_j^2]=\mu_e$；对于 $x\sim\mathcal{X}_j$，有 $\mathbb{E}_{x\sim\mathcal{X}_j}[x^Tx]=\Sigma_j$，故上式
$$
\begin{aligned}
&=\mathbb{E}_{(\theta_j,\epsilon_j^2)\sim\Theta}\left[\epsilon_j^2\right]\tr\left[\mathbb{E}_{x\sim\mathcal{X}_j}\left[x^Tx\right]\mathbb{E}_{X_j\sim\mathcal{X}_j}\left[\left(X_j^TX_j\right)^{-1}\right]\right]\\
&=\mu_e\tr\left[\Sigma_j\mathbb{E}_{X_j\sim\mathcal{X}_j}\left[\left(X_j^TX_j\right)^{-1}\right]\right]
\end{aligned}
$$

### 证明2

条件增强：$\mathcal{X}_j$ 为期望为 0 的正态分布

此时 $X_j^TX_j$ 满足 Inverse Wishart 分布，即
$$
\mathbb{E}_{X_j\sim\mathcal{X}_j}\left[\left(X_j^TX_j\right)^{-1}\right]=\frac{1}{n_j-D-1}Cov_j^{-1}
$$
$Cov_j$ 为 $\mathcal{X}_j$ 的协方差矩阵，其元素为 $\mathbb{E}[(x^k-\mathbb{E}[x^k])(x^l-\mathbb{E}[x^l])]=\mathbb{E}[x^kx^l]$，因此 $Cov_j=\Sigma_j$，故
$$
\begin{aligned}
\mu_etr\left[\Sigma_j\mathbb{E}_{X_j\sim\mathcal{X}_j}\left[\left(X_j^TX_j\right)^{-1}\right]\right]
&=\mu_e\tr\left[\Sigma_j\frac{1}{n_j-D-1}\Sigma_j^{-1}\right]\\
&=\frac{\mu_e}{n_j-D-1}\tr\left[I_D\right]\\
&=\frac{\mu_e}{n_j-D-1}D
\end{aligned}
$$

### 证明3

条件增强：场景为均值估计

- $D=1$，$x=X_j=1$
- $\Sigma_j=\mathbb{E}_{x\sim\mathcal{X}_j}[x^2]=1$
- $X_j^TX_j=n_j$

故
$$
\mu_etr\left[\Sigma_j\mathbb{E}_{X_j\sim\mathcal{X}_j}\left[\left(X_j^TX_j\right)^{-1}\right]\right]=\frac{\mu_e}{n_j}
$$


## 联邦学习模型

- 统一联邦
  $$
  \hat{\theta}_j^f=\frac1N\sum_{i=1}^M\hat\theta_i\cdot n_i
  $$
  其中 $N=\sum_{i=1}^Mn_i$
  
- 细粒度联邦
  $$
  \hat\theta_j^v=\sum_{i=1}^Mv_{ji}\hat\theta_i
  $$
  其中 $\sum_{i=1}^Mv_{ji}=1$



## 细粒度联邦的 MSE 的期望

### 结论

对于线性回归，$n_j$ 个样本的细粒度联邦的 MSE 的期望 $\mathbb{E}[(x\hat\theta_j^v-x\theta_j)^2]$ 为
$$
L_j+\left(\sum_{i\neq j}v_{ji}^2+\left(\sum_{i\neq j}v_{ji}\right)^2\right)\cdot\sum_{d=1}^D\mathbb{E}_{x\sim\mathcal{X}_j}\left[\left(x^d\right)^2\right]\cdot\sigma_d^2
$$
其中 $L_j$ 为
$$
\mu_e\sum_{i=1}^{M}v_{ji}^2\cdot\tr\left[\Sigma_j\mathbb{E}_{Y\sim\mathcal{D}(\theta_i,\epsilon_i^2)}\left[\left(X_i^TX_i\right)^{-1}\right]\right]
$$
如果 $\mathcal{X}_j$ 为均值为 0 的正态分布，$L_j$ 可简化为
$$
\mu_e\sum_{i=1}^Mv_{ji}^2\cdot\frac{D}{n_i-D-1}
$$
如果是对于均值估计，整个式子可简化为
$$
\mu_e\sum_{i=1}^Mv_{ji}^2\cdot\frac{1}{n_i}+\left(\sum_{i\neq j}v_{ji}^2+\left(\sum_{i\neq j}v_{ji}\right)^2\right)\cdot\sigma^2
$$

### 证明1

用 $\mathbb{E}_{Y\sim\mathcal{D}(\theta_i,\epsilon_i^2)}$ 表示 $i\in[M]$ 的数据的期望（Y为$\hat\theta_i$？？？）
$$
\begin{aligned}
(x\theta_j-x\hat\theta_j^v)^2
&=(x\theta_j-x\theta_j^v+x\theta_j^v-x\hat\theta_j^v)^2\\
&=(x\theta_j-x\theta_j^v)^2+(x\theta_j^v-x\hat\theta_j^v)^2+2(x\theta_j-x\theta_j^v)(x\theta_j^v-x\hat\theta_j^v)
\end{aligned}
$$
对于第三项，$\mathbb{E}_{Y\sim\mathcal{D}(\theta_j,\epsilon_j^2)}[x\theta_j^v-x\hat\theta_j^v]=0$

对于第二项，
$$
\begin{aligned}
(x\theta_j^v-x\hat\theta_j^v)^2
&=\left(x\sum_{i=1}^M v_{ji}\theta_i-x\sum_{i=1}^M v_{ji}\hat\theta_i\right)^2\\
&=\left(\sum_{i=1}^M v_{ji}x(\theta_i-\hat\theta_i)\right)^2
\end{aligned}
$$
$\theta_i\sim\Theta$ 相互独立，$X_i\sim\mathcal{X}_i$ 相互独立，因此 $\theta_i-\hat\theta_i$ 相互独立，故对于 $i\neq k$，
$$
v_{ji}x(\theta_i-\hat\theta_i)\cdot v_{jk}x(\theta_i-\hat\theta_i)
=v_{ji}x\mathbb{E}[\theta_i-\hat\theta_i]\cdot v_{jk}x\mathbb{E}[\theta_k-\hat\theta_k]
=0（？？？）
$$
因此上式
$$
\begin{aligned}
&=\sum_{i=1}^Mv_{ji}^2(x\theta_i-x\hat\theta_i)^2\\
&=\mu_e\sum_{i=1}^{M}v_{ji}^2\cdot\tr\left[\Sigma_j\mathbb{E}_{Y\sim\mathcal{D}(\theta_i,\epsilon_i^2)}\left[\left(X_i^TX_i\right)^{-1}\right]\right]
\end{aligned}
$$
对于第一项，
$$
\begin{aligned}
(x\theta_j-x\theta_j^v)^2
&=(x\theta_j-x\theta_j^v)^T(x\theta_j-x\theta_j^v)\\
&=(\theta_j-\theta_j^v)^Tx^Tx(\theta_j-\theta_j^v)\\
&=\tr\left[(\theta_j-\theta_j^v)^T\mathbb{E}_{x\sim\mathcal{X}_j}\left[x^Tx\right](\theta_j-\theta_j^v)\right]\\
&=\tr\left[\Sigma_j(\theta_j-\theta_j^v)(\theta_j-\theta_j^v)^T\right]
\end{aligned}
$$
考虑其内部项
$$
\begin{aligned}
\theta_j-\theta_j^v
&=\theta_j-\sum_{i=1}^Mv_{ji}\theta_i\\
&=(1-v_{jj})\theta_j-\sum_{i\neq j}v_{ji}\theta_i\\
&=\sum_{i\neq j}v_{ji}\theta_j-\sum_{i\neq j}v_{ji}\theta_i\\
&=\sum_{i\neq j}v_{ji}(\theta_j-\theta_i)
\end{aligned}
$$
又因为 $\theta_i\sim\Theta$ 独立同分布，即

- $\mathbb{E}[\theta_j\theta_j^T]=\mathbb{E}[\theta_i\theta_i^T]$
- $\mathbb{E}[\theta_j\theta_k^T]=\mathbb{E}[\theta_i\theta_l^T]$，$j\neq k$，$i\neq l$

因此
$$
\begin{aligned}
(\theta_j-\theta_j^v)(\theta_j-\theta_j^v)^T
&=\sum_{i\neq j}v_{ji}^2(\theta_j-\theta_i)(\theta_j-\theta_i)^T+\sum_{i,k\neq j,i\neq k}v_{ji}v_{jk}(\theta_j-\theta_i)(\theta_j-\theta_k)^T\\
&=\sum_{i\neq j}v_{ji}^2\left(\theta_j\theta_j^T-\theta_j\theta_i^T-\theta_i\theta_j^T+\theta_i\theta_i^T\right)+\sum_{i,k\neq j,i\neq k}v_{ji}v_{jk}\left(\theta_j\theta_j^T-\theta_j\theta_k^T-\theta_i\theta_j^T+\theta_i\theta_k^T\right)\\
&=\left(2\sum_{i\neq j}v_{ji}^2+\sum_{i,k\neq j,i\neq k}v_{ji}v_{jk}\right)\left(\mathbb{E}[\theta_j\theta_j^T]-\mathbb{E}[\theta_j\theta_i^T]\right)\\
&=\left(\sum_{i\neq j}v_{ji}^2+\left(\sum_{i\neq j}v_{ji}\right)^2\right)\left(\mathbb{E}[\theta_j\theta_j^T]-\mathbb{E}[\theta_j\theta_i^T]\right)
\end{aligned}
$$
对于矩阵 $\mathbb{E}[\theta_j\theta_j^T]-\mathbb{E}[\theta_j\theta_i^T]$

- 主对角线元素为 $\mathbb{E}[(\theta_j^d)^2]-\mathbb{E}[\theta_j^d]\mathbb{E}[\theta_i^d]=\mathbb{E}[(\theta_j^d)^2]-\mathbb{E}[\theta_j^d]^2=Var(\theta_j^d)=\sigma_d^2$
- 其他元素为 $\mathbb{E}[\theta_j^k\theta_j^l]-\mathbb{E}[\theta_j^k\theta_i^l]=\mathbb{E}[\theta_j^k]\mathbb{E}[\theta_j^l]-\mathbb{E}[\theta_j^k]\mathbb{E}[\theta_j^l]=0$（假设不同分量间相互独立）

即 $\mathbb{E}[\theta_j\theta_j^T]-\mathbb{E}[\theta_j\theta_i^T]=diag(\sigma_1^2,\sigma_2^2,\dots\sigma_D^2)$，故
$$
\begin{aligned}
\tr\left[\Sigma_j(\theta_j-\theta_j^v)(\theta_j-\theta_j^v)^T\right]
&=\left(\sum_{i\neq j}v_{ji}^2+\left(\sum_{i\neq j}v_{ji}\right)^2\right)\tr\left[\Sigma_j\left(\mathbb{E}[\theta_j\theta_j^T]-\mathbb{E}[\theta_j\theta_i^T]\right)\right]\\
&=\left(\sum_{i\neq j}v_{ji}^2+\left(\sum_{i\neq j}v_{ji}\right)^2\right)\cdot\sum_{d=1}^D\mathbb{E}_{x\sim\mathcal{X}_j}\left[\left(x^d\right)^2\right]\cdot\sigma_d^2
\end{aligned}
$$
综上，得证

### 证明2

条件增强：$\mathcal{X}_j$ 为期望为 0 的正态分布

由之前的结论，显然成立

### 证明3

条件增强：场景为均值估计

由 $\mathbb{E}_{x\sim\mathcal{X}_j}[x^2]=1$ 以及之前的结论，显然成立



## 统一联邦的 MSE 的期望

### 结论

对于线性回归，$n_j$ 个样本的统一联邦的 MSE 的期望 $\mathbb{E}[(x\hat\theta_j^f-x\theta_j)^2]$ 为
$$
L_j+\frac{\sum_{i\neq j}n_i^2+\left(\sum_{i\neq j}n_i\right)^2}{N^2}\cdot\sum_{d=1}^D\mathbb{E}_{x\sim\mathcal{X}_j}\left[\left(x^d\right)^2\right]\cdot\sigma_d^2
$$
其中 $L_j$ 为
$$
\mu_e\sum_{i=1}^{M}\frac{n_i^2}{N^2}\cdot\tr\left[\Sigma_j\mathbb{E}_{Y\sim\mathcal{D}(\theta_i,\epsilon_i^2)}\left[\left(X_i^TX_i\right)^{-1}\right]\right]
$$
如果 $\mathcal{X}_j$ 为均值为 0 的正态分布，$L_j$ 可简化为
$$
\mu_e\sum_{i=1}^M\frac{n_i^2}{N^2}\cdot\frac{D}{n_i-D-1}
$$
如果是对于均值估计，整个式子可简化为
$$
\frac{\mu_e}{N}+\frac{\sum_{i\neq j}n_i^2+(N-n_j)^2}{N^2}\cdot\sigma^2
$$

### 证明

代入 $v_{ji}=\frac{n_i}{N}$，得
$$
\sum_{i\neq j}v_{ji}^2+\left(\sum_{i\neq j}v_{ji}\right)^2
=\frac{\sum_{i\neq j}n_i^2+\left(\sum_{i\neq j}n_i\right)^2}{N^2}
$$
且
$$
\sum_{i=1}^Mv_{ji}^2\cdot\frac{1}{n_i}=\sum_{i=1}^M\frac{n_i^2}{N^2}\cdot\frac1{n_i}=\frac{1}{N}
$$
