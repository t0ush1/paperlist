# Data-Sharing Markets: Model, Protocol, and Algorithms to Incentivize the Formation of Data-Sharing Consortia

## 1 introduction

- 提出一种模型+协议+算法，DSC（Data-Sharing-Consortia），激励数据共享联盟的形成

- motivation
  - 共享数据在时间和精力上是昂贵的
  - 没有明确的成本效益分析，参与者通常会选择不共享的“默认”决策
  - 例如：
    - 医疗护理：医疗机构共享数据对患者的好处是否会超过克服技术方面挑战的代价
    - 欺诈检测：金融机构共享数据所得的收益是否至少会超过他们的数据为他人创造的收益
- contribution
  - 模型：引入两种激励机制
    - 保证共享数据的回报高于不共享（sharing dominance）
    - 参与者根据对联盟的数据贡献比例获得补偿（entitlement stake）
  - 协议
    - 实现sharing dominance和entitlement stake激励
    - 鲁棒性：检测不良的数据输入
    - 可持续性：用参与者创造的收益中支付成本
  - 算法：从参与者的数据中提取相应的价值
    - 支付运营联盟的成本
    - 补偿数据共享（一种比shapley值更好的方法）
- pre-condition：
  - 共享场景：无外部激励或抑制
  - 数据集成：不关心集成数据的技术，假设数据都可以组合



## 2 setup and notation

- 参与者 $i\in N$，数据 $D_i$，基准（benchmark）$B_i$
- 数据驱动的任务 $\mathcal{T}$
  - 例如机器学习、信息检索
  - 生命周期
    - setup：接受数据 $D_i$，返回任务实例 $T=\mathcal{T}(D_i)$
    - query：接受查询 $x$，返回答案 $T(x)$
    - evaluate：接受一组查询 $\vec{x}$ 和基准 $B=\{x,y=T(x)\}$，返回报酬 $P(T,\vec{x},B)$
- 参与者目标：回答所有查询并从答案中获得最大价值 $v_i$ 
  - $v_i=V_i(P(T,\vec{x_i},B_i))$ 由价值函数（value function）决定
  - $v_i$ 为隐私，但参与者可以通过竞标（bid）$b_i$ 暗示自己的价值，当 $b_i=v_i$ 时，称竞标是可信的
- $v_i$ 为独立任务获得的价值，$v_i^P$ 为加入联盟获得的价值
  - 理论上数据的价值是 1+1>2 的，但实际上由于 dirty data 或 strategic input，可能是 1+1<2。本文算法可以检测这些情况来防止损害数据集，保证 $v_i^P-v_i\ge 0$
  - 参与者难以确定 $v_i^P-v_i$，这会阻碍联盟的形成。本文提出的模型可以告诉参与者该值，从而让他们决定是否参与联盟



## 3 sharing model

- incentives

  - sharing dominance：$v_i^P\ge v_i$
  - entitlement stake
    - 收益分配（revenue allocation）函数 $RA$
    - 正反馈，促进 data sustainability

- costs

  - 任务设置，$C_T(\mathcal{T}(D_i))$

  - 数据存储，$C_S(D_i)$

  - 任务查询，$C_Q(\vec{x})$

  - 任务评估，$C_E(\vec{x},T,B)$

  - 运行收益分配函数，$C_{RA}$

  - 组合数据集并设置任务，$C_T(\mathcal{T}(D^I))$

  - 不考虑运行协议和上传数据的成本

  - 不参与者的效用函数（utility function）
    $$
    \begin{aligned}u_i(\vec{x_i},D_i,B_i)=V(P((D_i),\vec{x_i}),B_i)-C_T(\mathcal{T}(D_i))-\\C_S(D_i)-C_Q(\vec{x})-C_E(\vec{x},T,B)\end{aligned}
    $$

  - 参与者的效用函数
    $$
    \begin{aligned}u_i^P(\vec{x}_i,D_i,B_i)=V(P((D^I),\vec{x}_i),B_i)+RA(R(\vec{b},\vec{x}))-\\C_S(D_i)-C_Q(\vec{x})-C_E(\vec{x},T,B)-C_P\end{aligned}
    $$

    - $C_P=C_{RA}+C_T(\mathcal{T}(D^I))$：共享协议的开销



## 4 forming data-sharing consortia

- KD1：将存储、查询、评估的成本推给参与者
  - 防止参与者因为不付费过度使用资源
  - 要求参与者仔细考虑哪些数据值得分享
- KD2：平台承担所有其他费用
  - $u_i^P\ge u_i\Longrightarrow v_i^G+RA+C_T\ge C_P$
  - 可提高报酬 $v_i^G$ 和数据补偿 $RA$
- 四个阶段：合同协定、发送信号、价值提取、价值分配



## 5 revenue allocation

- 收益分配的要求：在第 3 阶段时 $C_{RA}$ 应小于剩余收益
  - 平均分配：成本低、不能激励数据的可持续性
  - 基于 entitlement 的分配：实现了 entitlement stake 激励
- 基于 shapley 值的算法
- shapley 值不适用的场景
- entitlement stake 算法



## 6 robustness against bad inputs

- 不良输入（bad inputs）：可能破坏 $v_i^P\ge v_i$ 的输入



## 7 evaluation

- settings
  - 机器学习任务
  - environment：python，16GB MacBook Pro，256TB Chameleon Testbed
  - 对不确定收益的处理
    - 机器学习的收益函数通常是不确定的，相同数据不同模型产生的收益不同
    - 训练 10 个模型，在测试集的 10 个 bootstrap 样本上测试每个模型，对共 100 个结果分数取均值
  - datasets
    - 实验不依赖于特定的数据集。第一个数据集得出结论，第二个数据集验证结论
    - Census Income Dataset from UCI：33K 个样本，15 个属性，1 个预测任务（预测收入 >= 50K）
    - hotel booking dataset：120K 个样本，32 个属性，1 个预测任务（预测预定取消）
    - 对UCI数据集研究数据分布的影响
      - biased split：在训练模型上使用 What-If Tool 来了解哪些属性对预测任务影响最大，然后确保一些参与者具有高影响属性，其他参与者没有
      - even split：数据集在参与者之间均匀分割，每个参与者以相同分布共享等量数据
    - 任务设置
      - 对数据规范化和标准化
      - 超参数搜索：网格搜索
      - 10 折交叉验证（10份，每次取1份做测试集，其他做训练集，将10次的MSE取平均）
      - 模型：UCI数据集（随机森林）；酒店数据集（CatBoost）
  - baselines
    - 代替的收入分配策略
    - Shapley值，SHA
    - Shapley值TMC的截断蒙特卡罗近似，TMC
- 联盟什么时候是可行的？
- 算法是否能检测并删除不良输入？
- 算法和基于shapley值的方法实际差异在哪？



## 8 related work

- 联邦学习的激励机制
- 联邦数据库和数据联邦
- 数据管理研究
- 当今的数据市场
- data lakes、commons、cooperatives
- 联邦学习、同态加密、多方学习、区块链
- 法律框架



## 9 conclutions

- DSC 可促进更多数据共享联盟的形成，能够成功检测出错误输入，保护了参与者，并且使用了比Shapley值执行代价更小且同样准确的收入分配函数，更好地补偿了数据贡献