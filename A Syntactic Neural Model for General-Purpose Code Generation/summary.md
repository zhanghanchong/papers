#### A Syntactic Neural Model for General-Purpose Code Generation

##### 语法模型

主体思想在于解码一棵抽象语法树（Abstract Syntax Tree，AST），以深度优先的顺序，每个时间步解码一个语法动作。语法动作分为APPLYRULE和GENTOKEN两种。令自然语言输入为$x$，生成代码段对应的AST为$y$，于是
$$
p(y | x) = \prod_{t = 1}^T p(a_t | x, a_{< t})
$$

* Unary Closure

  语法中可能存在一条一元扩展链，将该链压缩为一条语法可以减少解码步数，但会增大预测语法的数量，故为可选项。

* GENTOKEN

  生成的具体值可能是单token或多tokens的，故使用\</n\>作为标志结束的特殊token 。

##### 模型架构

* Encoder

  使用双向LSTM计算问题中$n$个单词的表示，其中第$i$个单词$w_i$对应的上下文表示为$h_i$。

* Decoder
  $$
  s_t = f_{\rm LSTM}([a_{t - 1}; c_t; p_t; n_{f_t}], s_{t - 1})
  $$
  $a_{t - 1}$是前一时刻动作的embedding，$c_t$是通过注意力机制得到的上下文向量，$p_t$是父节点动作信息的编码向量（$s_{p_t}$和$a_{p_t}$拼接），$n_{f_t}$是当前待扩展节点的embedding。
  $$
  \begin{aligned}
  p(a_t = {\rm APPLYRULE}[r] | x, a_{< t}) & = {\rm softmax}(W_R \cdot \tanh(W_1 \cdot s_t + b_1)) \\
  p(a_t = {\rm GENTOKEN}[v] | x, a_{< t}) & = p({\rm gen} | x, a_{< t}) \cdot p(v | {\rm gen}, x, a_{< t}) + p({\rm copy} | x, a_{< t}) \cdot p(v | {\rm copy}, x, a_{< t})
  \end{aligned}
  $$
  其中$p({\rm gen} | x, a_{< t})$和$p({\rm copy} | x, a_{< t})$由${\rm softmax}(W_S \cdot s_t)$计算得到。
  $$
  \begin{aligned}
  p(v | {\rm gen}, x, a_{< t}) & = {\rm softmax}(W_G \cdot \tanh(W_2 \cdot [c_t; s_t] + b_2)) \\
  p(w_i | {\rm copy}, x, a_{< t}) & = \frac{\exp(\omega(h_i, s_t, c_t))}{\sum_{j = 1}^n \exp(\omega(h_j, s_t, c_t))}
  \end{aligned}
  $$
  其中$\omega(\cdot)$是只有一个隐含层的网络。

##### 实验结果

* performance达到SOTA，语法规则建模起到重要作用。
* 与SEQ2TREE相比，生成AST的节点数较少，而且不会出现语法错误的情况，这两点使得模型优于SEQ2TREE。
* Frontier Node Embedding一般而言是有利于performance的。
* Parent Feeding在AST较大的情况下作用更明显。
* 错误预测主要出自错误的copy和功能实现不完整。

##### 未来工作

* 生成复杂嵌套结构的代码准确率有待提高。
* 更精确的evaluation方法，避免误判不匹配但实际等价的代码。
