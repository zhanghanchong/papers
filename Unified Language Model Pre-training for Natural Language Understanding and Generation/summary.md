#### Unified Language Model Pre-training for Natural Language Understanding and Generation

##### 模型架构

![model](asset/model.png)

通过共享参数以及不同的Attention Mask，融合Bidirectional、Left-to-Right、Right-to-Left、Sequence-to-Sequence语言模型。

##### 预训练任务

与BERT相同，采用MLM，对于Bidirectional还采用NSP。

##### 微调任务

* NLU：使用[SOS]分类；GLUE Benchmark，Extractive QA。
* NLG：Abstractive Summarization，Question Generation，Generative QA，Dialog Response Generation。

##### 实验结果

* performance超越大多数previous work。
* 基于Question Generation生成的问题有助于提升QA。

##### 未来工作

* Ablation Experiments。
* 目前只探索了单语言，可以尝试探索跨语言。
* 多任务学习。
