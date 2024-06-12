# Learned Cardinality Benchmark

This benchmark is proposed for verifying leraned cardinality. A brief description below introduces the background of cardinality estimation and details of using the benchmark.

## Background

Cardinality estimation plays a significant role in query optimization of database systems. It aims at estimating the size of sub-plans of each query and guiding the optimizer to select the optimal join operations. Performance of cardinality estimation has great impact on the quality of the generated query plans.

Recent works have explored how machine learning can be adopted into cardinality estimation. Like [this paper](https://arxiv.org/pdf/1809.00677.pdf), it proposes a multi-set MLP-based architecture (MSCN) to frame the cardinality estimation problem as a typical deep  learning task.  Despite the promising results, learned cardinality has a drawback that it neglacts the internal semantic logic of each query as a result of encoding queries into numerical vectors.

## Benchmark

We will first describe the model we use in our benchmark and then the design of specifications as well as instructions of using this benchmark.

### Network

We leverage the settings for the original MSCN except for not using a bitmap in our input. Excluding the bitmap makes the MSCN model flexible to various query inputs and largely reduce the model size.  Every query is featurized as a vector:

![](./resources/input_encoding.png)

where binary values stand for one-hot encoding of tables, joins, and queries present in the query. And the decimals such as `0.72' in the  figure is the normalized attribute value. 

The featurized query is further fed into a multi-set MLP-based model:

<div align=center><img width="300" height="300" src="./resources/model_architecture.png"/></div>

In this benchmark, we provide two trained networks with hidden size as 50 and 100 respectively, in directory [nets](./nets).

### Specifications

Our specifications are designed following the intuition of verifying internal logistics of the query. For example, if we have Table $t$ and query $q$, a naive specification can be

- $0\leq Cardinality(q)\leq $#total samples of $t$.

Intermediate specifications verify the consistence between estimations:

> $q_1$=SELECT COUNT(*) FROM title t WHERE t.prod_year>2015
>
> $q_2$=SELECT COUNT(*) FROM title t WHERE t.prod_year>2020
>
> $Cardinality(q_1)\geq Cardinality(q_2)$

More designed specifications that write in vnnlib format can be found in directory [specs](./specs/).

Our benchmark provides pairs of (i) learned cardinality (trained MSCN with different hidden size) and (ii) corresponding specifications, so that covers varied difficulties for verification tools.

## Verification Instructions

