#### RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers

##### Relation-Aware Self-Attention

对于一组输入$X = \{x_i\}_{i = 1}^n$，其中$x_i \in \R^{d_x}$，经过典型Transformer Encoder会得到
$$
\begin{aligned}
e_{i j}^{(h)} & = \frac{x_i W_Q^{(h)} (x_j W_K^{(h)})^T}{\sqrt{\frac{d_z}{H}}} \\
\alpha_i^{(h)} & = \rm{softmax}(e_i^{(h)}) \\
z_i^{(h)} & = \sum_{j = 1}^n \alpha_{i j}^{(h)} (x_j W_V^{(h)}) \\
z_i & = [z_i^{(1)}; \cdots; z_i^{(H)}] \\
\tilde{y}_i & = \rm{LayerNorm}(x_i + z_i) \\
y_i & = \rm{LayerNorm}(\tilde{y}_i + \rm{FC}(\rm{ReLU}(\rm{FC}(\tilde{y}_i))))
\end{aligned}
$$
$H$为attention head数量。为了表现数据库schema中原有的关系，采用类似于relative position的方法，
$$
\begin{aligned}
e_{i j}^{(h)} & = \frac{x_i W_Q^{(h)} (x_j W_K^{(h)} + r_{i j}^K)^T}{\sqrt{\frac{d_z}{H}}} \\
z_i^{(h)} & = \sum_{j = 1}^n \alpha_{i j}^{(h)} (x_j W_V^{(h)} + r_{i j}^V)
\end{aligned}
$$
考虑$R$个关系特性$\mathcal{R}^{(s)} \in X \times X (1 \le s \le R)$，于是$r_{i j}^K = r_{i j}^V = [\rho_{i j}^{(1)}; \cdots; \rho_{i j}^{(R)}]$，其中对于$\rho_{i j}^{(s)}$，如果$(i, j) \in \mathcal{R}^{(s)}$，那么其为一个可学习的embedding，否则为零向量。

##### 模型架构

* 问题定义

  令$Q = q_1 \cdots q_{|Q|}$为自然语言问题，$S = \langle \mathcal{C}, \mathcal{T} \rangle$为数据库schema，其中$\mathcal{C} = \{c_1, \cdots, c_{|\mathcal{C}|}\}$（$c_i = c_{i, 1} \cdots c_{i, |c_i|}$）为column集合，$\mathcal{T} = \{t_1, \cdots, t_{|\mathcal{T}|}\}$（$t_i = t_{i, 1} \cdots t_{i, |t_i|}$）为table集合。每个column有一个类型$\tau \in \{\rm{number}, \rm{text}\}$。定义schema对应的有向图$\mathcal{G} = \langle \mathcal{V}, \mathcal{E} \rangle$，其中$\mathcal{V} = \mathcal{C} \bigcup \mathcal{T}$，节点以其名字作为标签（对于column还需前置其类型），边集$\mathcal{E}$的定义如下图。

  ![edge](asset/edge.png)

  尽管该有向图包含了schema的所有信息，它不足以为以$Q$为上下文背景的未见过的schema编码，故定义图$\mathcal{G}_Q = \langle \mathcal{V}_Q, \mathcal{E}_Q \rangle$，其中$\mathcal{V}_Q = \mathcal{V} \bigcup Q, \mathcal{E}_Q = \mathcal{E} \bigcup \mathcal{E}_{Q \leftrightarrow S}$。
  
* 编码

  对于column和table，其初始表示$c_i^{\rm{init}}, t_i^{\rm{init}}$来自于

  * 对每个单词获取预训练好的embedding；
  * 对于多单词标签，将上述embedding用双向LSTM处理。

  同样用一个单独的双向LSTM处理问题$Q$，得到问题的初始表示$q_i^{\rm{init}}$。于是relation-aware self-attention的输入为
  $$
  X = (c_1^{\rm{init}}, \cdots, c_{|\mathcal{C}|}^{\rm{init}}, t_1^{\rm{init}}, \cdots, t_{|\mathcal{T}|}^{\rm{init}}, q_1^{\rm{init}}, \cdots, q_{|\mathcal{Q}|}^{\rm{init}})
  $$
  可选项：使用BERT架构处理$X$，使用BERT最终的表示作为每一项的初始表示。

* Schema Linking

  * Name-Based Linking

    显然如果问题中的部分单词可以和column或table名匹配，两者可以建立关系。具体而言，对于问题中所有长度在1到5之间的n-grams，确定其是否完美或部分匹配于某个column或table，然后确定是否向边集$\mathcal{E}_{Q \leftrightarrow S}$添加边QUESTION-COLUMN-M、QUESTION-TABLE-M、COLUMN-QUESTION-M、TABLE-QUESTION-M，其中M为EXACTMATCH、PARTIALMATCH、NOMATCH之一。

  * Value-Based Linking

    
