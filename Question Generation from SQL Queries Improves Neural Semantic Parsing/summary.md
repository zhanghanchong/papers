#### Question Generation from SQL Queries Improves Neural Semantic Parsing

##### 研究动机

探究模型性能与训练数据规模的关系，寻找使用更少的监督数据训练性能更优秀的模型的方法。

##### 语义解析模型

模型在WikiSQL数据集上进行实验。编码器使用双向GRU对问题进行编码，拼接双向最终hidden states作为解码器GRU的初始hidden state。解码器每个时间步生成SQL关键词、column、value中的一项。生成SQL关键词时，$e_i^{sql}$为SQL关键词对应的embedding。
$$
p_w^{sql}(i) = {\rm softmax}_i (W_{sql} [h_t^{dec}; e_i^{sql}])
$$
生成column时，$h_i^{col}$为column根据双向GRU编码得到的向量，$h_j^{cell}$为cell value根据双向GRU编码得到的向量，$\alpha_j^{cell}$为cell value对应的重要性分数，根据co-occurred question words的数量并进行softmax得到。
$$
p_w^{col}(i) = {\rm softmax}_i (W_{col} [h_t^{dec}; h_i^{col}; \sum_{j \in col_i} \alpha_j^{cell} h_j^{cell}])
$$
生成value时，$\hat{p}_w^{cell}(\cdot)$的计算方式与$p_w^{sql}(\cdot)$类似，超参数$\lambda$根据验证集进行调整。
$$
p_w^{cell}(j) = \lambda \hat{p}_w^{cell}(j) + (1 - \lambda) \alpha_j^{cell}
$$

##### 问题生成模型

随机采样SQL查询语句$x$，然后使用问题生成模型获取对应的问题，进行数据扩增。编码器使用双向GRU对SQL查询语句进行编码，第$i$个单词的编码向量为$h_i$，拼接双向最终hidden states得到$h_x$，以此作为解码器GRU的初始hidden state。在解码器中，对于第$t$个时间步，$s_t$为hidden state，$c_t$为根据注意力机制得到的上下文向量，$y_{t - 1}$为上一个预测单词的embedding。
$$
s_t = GRU(s_{t - 1}, y_{t - 1}, c_t)
$$
使用拷贝机制生成当前单词$y_t$，$\psi_g(\cdot)$为从词表$\mathcal{V}$中生成单词的分数，$\psi_c(\cdot)$为从SQL查询语句中拷贝单词的分数。
$$
\begin{aligned}
p(y_t | y_{< t}, x) & = \frac{e^{\psi_g(y_t)} + e^{\psi_c(y_t)}}{\sum_{v \in \mathcal{V}} e^{\psi_g(v)} + \sum_{w \in x} e^{\psi_c(w)}} \\
\psi_g(y_i) & = v_i^T W_g s_t \\
\psi_c(y_i) & = \tanh(h_i^T W_c) s_t
\end{aligned}
$$
其中$v_i$为one-hot向量。为了使得生成的问题语句具备多样性，模型中使用latent variable $z \sim \mathcal{N}(0, I_n)$，且在后验分布中，

$$
\begin{aligned}
\mu & = W_{\mu} [h_x; h_y] + b_{\mu} \\
\log(\sigma^2) & = W_{\sigma} [h_x; h_y] + b_{\sigma} \\
D_{KL}(Q(z | x, y) || p(z)) & = - \frac{1}{2} \sum_{j = 1}^n (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)
\end{aligned}
$$
其中$h_x$和$h_y$分别是source和target的编码结果，拼接$h_x$和$z$作为解码器的初始hidden state。由于模型会倾向于迫使KL散度为0，因此在训练时针对KL散度乘以一个变化的权值。

##### 实验结果

* 模型性能与数据规模的对数成正比。
* 数据扩增明显提升了模型性能。
* 引入latent variable有利于提升扩增数据的质量和多样性。
* 扩增数据中，大约27%的问题语句错误地表达了对应的SQL语句的含义，其中大部分遗漏了WHERE子句中的信息。其余73%的数据样例可以从两个方向考虑优化，一个是令问题生成模型考虑诸如列类型的更多表信息，一个是向问题生成模型中加入常识。
