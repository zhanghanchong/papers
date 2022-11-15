#### PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models

##### PICARD方法

PICARD是自回归语言模型在生成过程中可以自由加入的技术方法，在模型每一步生成token时，模型首先选取概率top-$k$的tokens，然后通过检查拒绝其中不合法的tokens，从而达到限制输出的目的，降低输出结果的语法错误。PICARD有四种模式（等级从低到高）：

* off：无检查，即不使用PICARD。
* lexing：限制生成的tokens为SQL关键词、标点符号、运算符、SQL条件值、schema项，检测关键词拼写错误，拒绝非法表或列。
* parsing without guards：尝试将模型输出解析成一个表示AST的数据结构，拒绝非法结构（例如检测子句缺失或顺序错误），拒绝tid.cid（或者alias.cid，tid as alias）但是cid不在tid中的情况，禁止重复绑定表别名。
* parsing with guards：确保相应的表出现在from子句中，如果匹配到单独的cid，则确保from子句中有恰好一个包含其的表。

##### 实验结果

* T5-3B+PICARD性能达到SOTA，超过LGESQL+ELECTRA。
* T5-Base+PICARD性能能够与T5-Large相当，T5-Large+PICARD性能能够与T5-3B相当。
* 使用PICARD后，生成出不可执行SQL的概率大大下降。
* 使用PICARD的off或lexing模式时，beam size对性能几乎没有影响，使用更高级的模式后，beam size取4时性能基本饱和；$k$的影响不大，取2即可，避免使beam search退化成greedy search。
* 仅在模型预测的最后一步使用PICARD，性能会与每一步均使用PICARD相比有差距，这种差距会随着beam size增大而缩小。
