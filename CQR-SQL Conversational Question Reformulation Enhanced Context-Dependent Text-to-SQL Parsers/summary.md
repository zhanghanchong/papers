#### CQR-SQL: Conversational Question Reformulation Enhanced Context-Dependent Text-to-SQL Parsers

##### 模型方法

![pipeline](asset/pipeline.png)

![rewrite](asset/rewrite.png)

![model](asset/model.png)

首先进行改写，然后分别利用改写前和改写后的语句进行编码解码，通过两边的一致性优化模型。

##### 实验结果

![result](asset/result.png)