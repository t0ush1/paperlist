# A marketplace for data: An algorithmic solution

## Introduction

- 设计一个实时数据市场，为训练数据定价，匹配买家卖家

### motivation

- 数据市场为什么现在产生
  - 机器学习在工业中占据重要地位，公司发展预测能力是有必要的
  - 刚起步的公司难以获得用于训练的数据，需要购买
- 数据作为资产的特殊性
  - 可零边际成本（额外生产一单位产品所需的成本）复制
  - 价值具有组合性：特定数据集对企业的价值取决于其他潜在相关的数据集，难以定价
  - 预测任务和提高准确度的价值在不同场景差别很大：预测准确度增加 10% 对于金融和物流有不同的价值
  - 如果不首先将数据应用于预测任务，很难先验地验证数据的真实性和有用性：卫星图像的数据集可能对金融有很强的预测性，但对物流可能没有用处
- 目前的在线市场为什么无法满足
  - 传统实时市场：在线广告拍卖和预测市场
  - 传统商品不可复制，可以二次价格拍卖，买家对商品价值有先验等

### contribution

- 双边数据市场模型；关键挑战（key challenges）的形式化定义
  - 激励买家真实地报告其内部估值
  - 更新一组相关数据集的价格，使其收入随时间推移而最大化
  - 将收入按训练特征（training features）公平分配，使卖家根据边际贡献（固定成本+利润）获得报酬
  - 构建实现上述功能的可高效计算的算法
- 算法解决方案；理论保证
  - 算法过程：将训练特征分配给买家并获取收入；更新该特征的售价；在卖家间分配收入
  - 关键贡献：对于可自由复制商品的合作博弈中“公平”的概念，其推广了 Shapley 公平；一个可信的、无悔的用于拍卖特定类别组合商品的机制，基于 Myerson's payment function 和 Multiplicative Weights algorithm
- two-sided：对于一个每次交易收费的平台，向买家和卖家分别收取 a 和 b 的费用，如果交易量仅依赖于总费用 a+b，则是 one-sided，否则是 two-sided

### 库存优化（Inventory Optimization）的例子

- 卖家：零售商店；出售匿名的每分钟步行交通数据流
- 买家：物流公司；预测未来库存需求，需要各种时间序列数据
  - 不清楚哪些步行交通数据流最具有预测性，并且在预算范围内
  - 具有成本模型：10% 的容量过剩的成本是每周 \$10000
  - 向数据市场投标：预测准确度比上周每提高一个百分比就支付 \$1000
- 数据市场的步骤：
  1. 物流公司提供预测任务（历史库存需求的时间序列）和一个投标
  2. 市场提供步行数据，根据买家投标和该数据的当前价格确定本次出售价格
  3. 根据步行数据和历史库存需求数据拟合 ML 模型
  4. 收入仅基于库存需求预测准确度的提高量
  5. 收入分配给所有提供数据的卖家
  6. 步行数据相关的价格被更新
- 发现该例子可以适应其他各种商业环境，因此作者认为上述过程同样适用于其他 ML 任务的数据，而不需要先验地知道什么数据组合是有用的



## The Model - Participants and Dynamics

- $M$ 个卖家
  - 特征 $X_j\in \mathbb{R}^T,j\in [M]$
  - 子特征矩阵 $\boldsymbol{X}_S,S\subset [M]$
- $N$ 个买家
  - 预测任务 $Y_n\in \mathbb{R}^T, n\in[N]$
  - 预测收益函数 $\mathcal{G}_n:\mathbb{R}^{2T}\rightarrow[0,1]$
    - 输入：预测任务 $Y_n$ 和估计值 $\hat{Y}_n$
    - 输出：预测的质量，如 1-RMSE 或 Accuracy
    - 认为所有 $\mathcal{G}_n$ 相同，$\mathcal{G}=\mathcal{G}_n$
  - 准确度估值 $\mu_n\in\mathbb{R}_+$
    - 买家愿意为 $\mathcal{G}$ 每提高一单位所支付的价格
    - 总价值 $\mu_n\cdot \mathcal{G}(Y_n,\hat{Y}_n)$
    - 买家对数据的估值不取决于特定数据集，而是预测准确性的提高
  - 向市场公开投标 $b_n\in\mathbb{R}_+$
- 市场（买家一次一个进入市场）
  - 当前价格 $p_n\in\mathbb{R}_+$
  - 机器学习/预测算法 $\mathcal{M}:\mathbb{R}^{MT}\to\mathbb{R}^T$
    - 输入：出售的特征 $\boldsymbol{X}_M$
    - 输出：估计值 $\hat{Y}_n$
    - 可由市场或买家提供

  - 分配函数 $\mathcal{AF}:(p_n,b_n;\boldsymbol{X}_M)\rightarrow\widetilde{\boldsymbol{X}}_M,\widetilde{\boldsymbol{X}}_M\in\mathbb{R}^M$
    - 输入：当前价格 $p_n$，投标 $b_n$
    - 输出：分配的特征 $\widetilde{\boldsymbol{X}}_M$

  - 收益函数 $\mathcal{RF}:(p_n,b_n,Y_n;\mathcal{M},\mathcal{G},\boldsymbol{X}_M)\to r_n,r_n\in\mathbb{R}_+$
    - 输入：当前价格 $p_n$，投标 $b_n$，预测任务 $Y_n$
    - 输出：从买家提取的收益 $r_n$

  - 收益分配函数 $\mathcal{PD}:(Y_{n},\widetilde{\boldsymbol{X}}_{M};\mathcal{M},\mathcal{G})\rightarrow\psi_{n},\psi_{n}\in[0,1]^{M}$
    - 输入：预测任务 $Y_n$，分配的特征 $\widetilde{\boldsymbol{X}}_{M}$
    - 输出：每个特征的边际价值 $\psi_{n}$

  - 价格更新函数 $\mathcal{PF}:(p_{n},b_{n},Y_{n};\mathcal{M},\mathcal{G},\boldsymbol{X}_{M})\rightarrow p_{n+1},p_{n+1}\in\mathbb{R}_{+}$
    - 输入：当前价格 $p_n$，投标 $b_n$，预测任务 $Y_n$
    - 输出：下一个买家的价格 $p_{n+1}$

  - 买家效用函数  $\mathcal{U}:\mathbb{R}_+\times \mathbb{R}^T\rightarrow \mathbb{R}$
    - $\mathcal{U}(b_n,Y_n):=\mu_n\cdot\mathcal{G}(Y_n,\hat{Y}_n)-\mathcal{RF}(p_n,b_n,Y_n)$
    - $\hat{Y}_{n}=\mathcal{M}(Y_{n},\widetilde{\boldsymbol{X}}_{M})$
    - $\widetilde{\boldsymbol{X}}_{M}=\mathcal{AF}(p_{n},b_{n};\boldsymbol{X}_{M})$

- 市场动态
  - 随机初始化 $p_0,b_0,Y_0$
  - 对于每个 $n\in [N]$
    1. 设置 $p_n=\mathcal{PF}(p_{n-1},b_{n-1},Y_{n-1})$
    2. 买家 $n$ 提供预测任务 $Y_n$
    3. 买家 $n$ 提供投标 $b_n=\text{arg max}_{z\in\mathbb{R_+}}\mathcal{U}(z,Y_n)$
    4. 市场分配特征 $\widetilde{\boldsymbol{X}}_{M}=\mathcal{AF}(p_n,b_n;\boldsymbol{X}_M)$ 给买家 $n$
    5. 买家 $n$ 得到预测收益 $\mathcal{G}\Big(Y_n,\mathcal{M}(\widetilde{X}_M)\Big)$
    6. 市场从买家 $n$ 提取价值 $r_n=\mathcal{RF}(p_n,b_n,Y_n;\mathcal{M},\mathcal{G})$
    7. 市场根据 $\psi_{n}=\mathcal{PD}(Y_{n},\widetilde{X}_{M};\mathcal{M},\mathcal{G})$ 分配收益 $r_n$

  - 特性
    - 买家不会获得潜在特征，防止卖家不愿意发布有潜在价值的数据流（数据是可自由复制的，卖家无法控制谁之后会访问这些数据）
    - 因为是市场定价，所以等价于单个卖方向市场提供多个数据流，通过调整 $p_n$ 获得最大收入
    - 买家 $n$ 只进入市场一次，并获得评估后离开。不考虑多阶段效用的复杂情况
    - 不考虑数据集被多次复制后的外部效用（一家公司对数据集的效用可能依赖于之后其他公司对该数据集的访问）


