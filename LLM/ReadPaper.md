# LLM + RL的理论 论文

*Created by KennyS*

---

## AI models collapse when trained on revursively generated data

**分析**

1. 使用数据训练模型, 再生成伪数据继续训练....
    - 高概率事件会被高估
    - 低概率事件会被低估

2. 数据的分布会发生变化, 结果也会发生变化
    - 同样应当关心数据的质量
    - 不断地递归模型, 模型会遗忘数据中低概率的事件。模型的输出会越来越窄..
    - PPL的图：更长的尾部, 后迭代的模型开始生成原始模型永远不会生成的样本

3. 实验设计

    - 控制变量：no data preserved vs 10% data preserved, 是否保留10%真实数据
    - metrics: PPL
    $$
    L = -\frac{1}{N} \sum_{i}^{N} log P(y_{i}) \\
    PPL = exp(-\frac{1}{N} \sum_{i}^{N} log P(y_{i})) \\
    PPL = exp(L)
    $$

4. 现实世界中混入了大量aigc数据, 很难分辨

**错误来源**

1. 统计逼近的误差
    - 获取实际分布的时候, 在采样时本身就会导致误差
2. 函数表达性的误差
    - 模型表达能力有限
3. 函数逼近的误差
    - 模型没有训练的很好
