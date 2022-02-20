#### TRANX: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation

##### 模型架构

* Encoder

  使用双向LSTM对问题中每个单词$x_i$计算上下文表示$h_i$。

* Decoder

  与YN17模型相比，增加了扩展规则REDUCE，用于处理?和*的情况。
  $$
  s_t = f_{\rm LSTM}([a_{t - 1}; \tilde{s}_{t - 1}; p_t], s_{t - 1})
  $$
  $a_{t - 1}$是前一步动作的embedding，$\tilde{s}_t = \tanh(W_c [c_t; s_t])$，其中$c_t$是根据注意力机制利用$h$计算得到的上下文向量，$p_t$是frontier field $n_{f_t}$的embedding和$s_{p_t}$的拼接。
  $$
  \begin{aligned}
  p(a_t = {\rm APPLYCONSTR}[c] | a_{< t}, x) & = {\rm softmax}(a_c^T W \tilde{s}_t) \\
  p(a_t = {\rm GENTOKEN}[v] | a_{< t}, x) & = p({\rm gen} | a_{< t}, x) \cdot p(v | {\rm gen}, a_{< t}, x) + p({\rm copy} | a_{< t}, x) \cdot p(v | {\rm copy}, a_{< t}, x) \\
  p(x_i | {\rm copy}, a_{< t}, x) & = {\rm softmax}(h_i^T W \tilde{s}_t)
  \end{aligned}
  $$
  $p({\rm gen} | \cdot)$和$p({\rm copy} | \cdot)$由${\rm softmax}(W \tilde{s}_t)$计算得到，$p(v | {\rm gen}, \cdot)$的计算与生成constructor类似。

##### 实验结果

* 在多个数据集performance达到SOTA。

* Parent Feeding对于语法简单的数据集效果较差。
* 使用简单的answer pruning策略（去掉候选SQL中执行后无结果的查询语句）可以获得较大的performance提升。
