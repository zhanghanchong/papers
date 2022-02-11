#### SQLNet: Generating Structured Queries from Natural Language Without Reinforcement Learning

##### 模型架构

* Sketch-Based

  ![sketch](asset/sketch.png)

  通过sketch-based预测slot值的方式，避免order-matters的问题。

* Sequence-to-Set和Column-Attention

  预测column names的一个子集，用于条件子句中。令$H_Q$为一个$d \times L$（$d$为embedding size，$L$为问题的sequence length）的矩阵，其中$H^Q$的第$i$列$H_Q^i$代表问题的第$i$个token经过双向LSTM的hidden state。令$E_{col}$是column name的embedding，由双向LSTM计算得到。计算$H_Q$和$E_{col}$的两个双向LSTM不共享参数权值。
  $$
  \begin{aligned}
  v_i & = E_{col}^T W H_Q^i \\
  w & = \rm{softmax}(v) \\
  E_{Q | col} & = H_Q w \\
  P_{\rm{wherecol}}(col | Q) & = \sigma((u_a^{\rm{col}})^T \tanh(U_c^{\rm{col}} E_{col} + U_q^{\rm{col}} E_{Q | col}))
  \end{aligned}
  $$
  $W_{d \times d}$是一个可训练矩阵，$U_c^{\rm{col}}$和$U_q^{\rm{col}}$是两个$d \times d$的可训练矩阵，$u_a^{\rm{col}}$是一个$d$维的可训练向量。

* SQLNet

  * WHERE子句

    * Column Slots

      首先预测一个$K$值，然后选取$P_{\rm{wherecol}}(col | Q)$中的top-$K$作为WHERE子句中的$K$个columns。预设$K$的上界为$N$，于是对$K$的预测转化为$0$到$N$的$(N + 1)$分类问题。
      $$
      P_{\rm{\# col}}(K | Q) = \rm{softmax}(U_1^{\rm{\# col}} \tanh(U_2^{\rm{\# col}} E_{Q | Q}))_{K + 1}
      $$
      $U_1^{\rm{\# col}}$是一个$(N + 1) \times d$的可训练矩阵，$U_2^{\rm{\# col}}$是一个$d \times d$的可训练矩阵。

    * OP Slot

      预测OP是一个3（=，>，<）分类任务。
      $$
      P_{\rm{op}}(i | Q, col) = \rm{softmax}(U_a^{\rm{op}} \tanh(U_c^{\rm{op}} E_{col} + U_q^{\rm{op}} E_{Q | col}))_i
      $$
      $U_a^{\rm{op}}$是一个$3 \times d$的可训练矩阵，$U_c^{\rm{op}}$和$U_q^{\rm{op}}$是两个$d \times d$的可训练矩阵。
      
    * VALUE Slot
    
      使用sequence-to-sequence结构预测VALUE，Decoder使用Pointer Network。令$h$代表先前生成序列的hidden state。
      $$
      \begin{aligned}
      a(h)_i & = (u_a^{\rm{val}})^T \tanh(U_q^{\rm{val}} H_Q^i + U_c^{\rm{val}} E_{\rm{col}} + U_h^{\rm{val}} h) \\
      P_{\rm{val}}(i | Q, col, h) & = \rm{softmax}(a(h))_i
      \end{aligned}
      $$
      $U_q^{\rm{val}}$、$U_c^{\rm{val}}$、$U_h^{\rm{val}}$是三个$d \times d$的可训练矩阵，$u_a^{\rm{val}}$是一个$d$维的可训练向量。
    
  * SELECT子句
  
    * Column Slot
    
      预测SELECT column是一个$C$分类任务，$C$代表column的数量。
      $$
      \begin{aligned}
      sel_i & = (u_a^{\rm{sel}})^T \tanh(U_c^{\rm{sel}} E_{col_i} + U_q^{\rm{sel}} E_{Q | col_i}) \\
      P_{\rm{selcol}}(i | Q) & = \rm{softmax}(sel)_i
      \end{aligned}
      $$
      $U_c^{\rm{sel}}$和$U_q^{\rm{sel}}$是两个$d \times d$的可训练矩阵，$u_a^{\rm{sel}}$是一个$d$维的可训练向量。
    
    * Aggregator Slot
    
      预测aggregator是一个6（AVG，COUNT，MAX，MIN，SUM，NULL）分类任务。
      $$
      P_{\rm{agg}}(i | Q, col) = \rm{softmax}(U_a^{\rm{agg}} \tanh(U_q^{\rm{agg}} E_{Q | col}))_i
      $$
      $U_q^{\rm{agg}}$是一个$d \times d$的一个可训练矩阵，$U_a^{\rm{agg}}$是一个$6 \times d$的一个可训练矩阵。
  
* 损失函数

  大部分使用交叉熵损失函数。对于特殊的sequence-to-set，令$y_j \in \{0, 1\}$表示第$j$个column是否出现在WHERE子句中，于是$P_{\rm{wherecol}}$对应的损失函数为
  $$
  loss(col, Q, y) = - (\sum_{j = 1}^C \alpha y_j \log P_{\rm{wherecol}}(col_j | Q) + (1 - y_j) \log(1 - P_{\rm{wherecol}}(col_j | Q)))
  $$
  $\alpha$是一个用于平衡正负样本的超参数。

##### 评价指标

* Logical-form accuracy：生成的查询语句与ground truth的完美匹配率。
* Query-match accuracy：排除由于ordering导致的假阴性。
* Execution accuracy：执行结果准确率。

##### 实验结果

* 模型performance大幅超越Seq2SQL。
* 超越的主要部分在于WHERE子句，此外column attention也起到一定作用。
