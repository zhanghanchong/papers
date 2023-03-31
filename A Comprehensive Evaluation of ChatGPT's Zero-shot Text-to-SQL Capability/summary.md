#### A Comprehensive Evaluation of ChatGPT's Zero-shot Text-to-SQL Capability

##### 总体结论

* 相较于使用完整训练数据的SOTA模型，ChatGPT有很强的zero-shot能力，性能仅低14%。
* 在生成SQL语句时，ChatGPT有很强的鲁棒性，在Spider数据集的一些鲁棒性setting下性能仅比SOTA低7.8%。
* 在数据库列名被对抗性修改的ADVETA场景下，ChatGPT的性能比SOTA高4.1%。
* ChatGPT生成的SQL的EM分数很低，因为同样的逻辑可以有不同的SQL表达，因此主要使用EX指标进行评价。

##### 方法

![prompt](asset/prompt.png)

##### 数据集

* Spider
* Spider-SYN：使用同义词改写Spider的自然语言问句。
* Spider-DK：从Spider的dev集中采样来自10个数据库的535个问题SQL对，修改这些数据使得其中包含领域知识。
* Spider-Realistic：删除问句中明确提到的列名，包含508条数据。
* Spider-CG(SUB)：评估模型的组合泛化性，在不同的样例间进行子句替换。
* Spider-CG(APP)：评估模型的组合泛化性，将一条子句单独添加在另一条问句后面。
* ADVETA(rpl)：对抗地替换列名。
* ADVETA(add)：增加新列名。
* CSpider：将Spider数据集翻译成中文。
* DuSQL
* SParC
* CoSQL

##### 实验结果

* 在Spider-SYN和Spider-Realistic数据集上，ChatGPT与SOTA模型的性能差距增大了。
* 在Spider-DK和ADVETA数据集上，ChatGPT在需要额外知识或者增加了对抗修改的场景下表现出较强的鲁棒性。
* 在Spider-CG数据集上，zero-shot的ChatGPT有着较强的组合泛化性，在Spider-CG(SUB)数据集上表现甚至超过原版Spider。
* 在多轮对话数据集上，ChatGPT表现出较强的上下文建模能力，尽管总体性能与SOTA模型仍然有差距。
* 在中文数据集上，ChatGPT表现弱于英文数据集，说明其跨语言泛化能力仍然需要提升。
