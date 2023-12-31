# Trading Data in Good Faith: Integrating Truthfulness and Privacy Preservation in Data Markets

- 在数据市场中整合可信度和隐私保护
- motivation：
  - 用户希望在保护隐私的同时，确认服务商真实地收集和处理了数据
  - 真实性和隐私性是“矛盾”的
- contribution：
  - 提出TPDM模型（Truthfulness and Privacy preservation in Data Markets）
    - 同态加密+基于身份的签名
    - 批量验证+数据处理+结果验证，同时维护身份保护和数据机密性
  - 部署在雅虎音乐评分数据集上，并评估性能

- 一个场景：

  - 推特的企业 api 平台 Gnip 收集推特用户的社交媒体数据，深入挖掘对特定受众的洞察，并为 500强企业提供数据分析解决方案

  - 三方：
    - Gnip：service provider
    - 推特用户：data contributors
    - 500强企业：data consumers

- 风险

  - provider 为了利益最大化，可能做出投机或非法行为，影响数据真实性
    - 部分数据收集攻击：伪造或合成数据混合到原始数据集中
    - 部分数据处理攻击：没有完全处理原始数据而得到一个假的结果
  - provider 可能会泄露 contributors 隐私给 consumers
    - 即使隐藏 contributors 的真实身份，也不应将原始数据透露给 consumers

- 三个挑战

  - 验证数据收集的真实性和保护隐私是相互矛盾的目标（加密程度越高，真实性越低）
  - 一些 provider 只提供处理后的数据而不是原始数据，难以保证真实性
  - 数据市场需要高效运行，传统的验证数据身份和完整性的方法资源开销大