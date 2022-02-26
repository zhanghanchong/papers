#### Graph Attention Networks

##### 模型架构

* Graph Attentional Layer

  输入节点特征$h = \{\vec{h}_1, \vec{h}_2, \cdots, \vec{h}_N\}$，其中$\vec{h}_i \in \R^F$，输出新的节点特征$h' = \{\vec{h}'_1, \vec{h}'_2, \cdots, \vec{h}'_N\}$，其中$\vec{h}'_i \in \R^{F'}$。定义权值矩阵为$W \in \R^{F' \times F}$，定义注意力机制$a: \R^{F'} \times \R^{F'} \rightarrow \R$用于计算$e_{i j} = a(W \vec{h}_i, W \vec{h}_j)$。
  $$
  \alpha_{i j} = {\rm softmax}_j(e_{i j}) = \frac{\exp(e_{i j})}{\sum_{k \in \mathcal{N}_i} \exp(e_{i k})}
  $$
  在实验中，$a$为一个单层前向神经网络，参数化表示为一个权值向量$\vec{a} \in \R^{2 F'}$，并使用LeakyReLU非线性（取$\alpha = 0.2$）。
  $$
  \vec{h}'_i = \sigma(\sum_{j \in \mathcal{N}_i} \alpha_{i j} W \vec{h}_j)
  $$
  进一步，使用multi-head attention，
  $$
  \vec{h}'_i = {\rm Concat}_{k = 1}^K \sigma(\sum_{j \in \mathcal{N}_i} \alpha_{i j}^k W^k \vec{h}_j)
  $$
  这样得到的$h'$向量的维度为$K F'$，故对于最后一层则使用平均而非拼接，即
  $$
  \vec{h}'_i = \sigma(\frac{1}{K} \sum_{k = 1}^K \sum_{j \in \mathcal{N}_i} \alpha_{i j}^k W^k \vec{h}_j)
  $$

* 
