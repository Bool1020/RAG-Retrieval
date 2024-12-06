<h1 align="center">RAG-Retrieval</h1>
<p align="center">
    <a href="https://pypi.org/project/rag-retrieval/#description">
            <img alt="Build" src="https://img.shields.io/pypi/v/rag-retrieval?color=brightgreen">
    </a>
    <a href="https://www.pepy.tech/projects/rag-retrieval">
            <img alt="Build" src="https://static.pepy.tech/personalized-badge/rag-retrieval?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads">
    </a>
    <a href="https://github.com/NLPJCL/RAG-Retrieval">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/NLPJCL/RAG-Retrieval/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
</p>
<h4 align="center">
    <p>
        <a href="#使用统一的方式推理不同的RAG Reranker模型">使用统一的方式推理不同的RAG Reranker模型</a> |
        <a href="#微调全链路的RAG检索模型">微调全链路的RAG检索模型</a> |
        <a href="#实验结果">实验结果</a> |
        <a href="#license">License</a> 
    <p>
</h4>

RAG-Retrieval 提供了全链路的RAG检索微调(train)和推理(infer)以及蒸馏(distill)代码。
- 对于微调，**支持微调任意开源的RAG检索模型**，包括向量（embedding、图a）、迟交互式模型（colbert、图d）、重排序模型（cross encoder(bert)、图c, llm-based reranker）。
- 对于推理，RAG-Retrieval专注于重排序(reranker)，开发了一个轻量级的python库[rag-retrieval](https://pypi.org/project/rag-retrieval/),**提供统一的方式调用任意不同的RAG排序模型**。
- 对于蒸馏，支持将基于LLM的reranker模型蒸馏到基于bert的reranker模型中。


![ColBERT](pictures/models.png)

# 社区交流

[加入我们微信群聊](https://www.notion.so/RAG-Retrieval-Roadmap-c817257e3e8a484b8850cac40a3fcf88)

# 最新更新

- 10/21/2024: RAG-Retrieval发布基于LLM做Reranker任务的两种不同方法，以及将其蒸馏到bert中的方法。[LLM在Reranker任务上的最佳实践？A simple experiment report（with code）](https://zhuanlan.zhihu.com/p/987727357)


- 6/5/2024: RAG-Retrieval的Embedding模型的MRL loss实现。[RAG-Retrieval：让MRL loss成为训练向量(embedding)模型的标配](https://zhuanlan.zhihu.com/p/701884479)

- 6/2/2024: RAG-Retrieval实现基于LLM偏好监督RAG检索器微调。[RAG-Retrieval实现基于LLM偏好监督RAG检索器微调](https://zhuanlan.zhihu.com/p/701215443)

- 5/5/2024:发布RAG-Retrieval的轻量级的python库[RAG-Retrieval：你的RAG应用值得更好的排序推理框架](https://zhuanlan.zhihu.com/p/692404995)

- 3/18/2024:发布RAG-Retrieval [RAG-Retrieval知乎介绍](https://zhuanlan.zhihu.com/p/683483778)

# Roadmap

- [RAG-Retrieval Roadmap](https://www.notion.so/RAG-Retrieval-Roadmap-c817257e3e8a484b8850cac40a3fcf88)



# 使用统一的方式推理不同的RAG Reranker模型

## 为什么要做Reranker模型的推理,甚至开发一个包？

排序模型是任何检索架构的重要组成部分，也是 RAG 的重要组成部分，但目前的现状是：

- 开源的排序模型很多，在A场景表现好的模型，在B场景不一定表现好，很难知道该使用哪一个。
- 另外，新的排序模型不断的出现，如今年3月份BGE才发布的LLM Reranker，使用decoder-only的大模型来对段落重排序，非常有前景。
- 所有不同的排序模型，都倾向于自己开发一套库来进行排序，这导致了更高的壁垒，新用户需要熟悉每一种排序模型的输入和输出，以及安装各种不同的依赖。

## rag-retrieval的特点

因此，RAG-Retrieval开发了一个轻量级的python库[rag-retrieval](https://pypi.org/project/rag-retrieval/),提供统一的方式调用任意不同的RAG排序模型，其有以下的特点。

- 支持多种排序模型：支持常见的开源排序模型(Cross Encoder Reranker,Decoder-Only 的LLM Reranker)

- 长doc友好：支持两种不同的对于长doc的处理逻辑(最大长度截断，切分取最大分值)。

- 益于扩展：如果有新的排序模型，用户只需要继承basereranker，并且实现rank以及comput_score函数即可。

## 安装环境
```bash
#为了避免自动安装的torch与本地的cuda不兼容，建议进行下一步之前先手动安装本地cuda版本兼容的torch。
pip install rag-retrieval
```

## 支持的Reranker模型

### Cross Encoder Reranker

对于Cross Encoder Reranker，只要其使用transformers的**AutoModelForSequenceClassification**，那么就可以使用rag_retrieval的Reranker来进行推理。举例如下。

- **bge系列的Cross Encoder模型，例如(BAAI/bge-reranker-base, BAAI/bge-reranker-large, BAAI/bge-reranker-v2-m3)**

- **bce的Cross Encoder模型，例如(maidalun1020/bce-reranker-base_v1)**


### LLM Reranker 

对于LLM Reranker，rag_retrieval的Reranker支持多种强大的LLM排序模型。也支持使用任意的LLM的chat模型来进行zero shot排序。举例如下。

- **bge系列的LLM Reranker模型，例如(BAAI/bge-reranker-v2-gemma, BAAI/bge-reranker-v2-minicpm-layerwise )**

## 使用

**rag-retrieval包详细的使用方法和注意事项可以参考[Tutorial](https://github.com/NLPJCL/RAG-Retrieval/blob/master/examples/Reranker_Tutorial.md)**

**我们做了大量的测试与下面原有推理框架对齐，详见[Tests](https://github.com/NLPJCL/RAG-Retrieval/tree/master/tests)，他们需要不同的模块来执行，rag_retrieval则使用了统一的接口。**

如 FlagEmbedding的 FlagReranker,FlagLLMReranker,LayerWiseFlagLLMReranker。

如 BCEmbedding 的 RerankerModel 


# 微调全链路的RAG检索模型

## 安装环境
```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
#为了避免自动安装的torch与本地的cuda不兼容，建议进行下一步之前先手动安装本地cuda版本兼容的torch。
pip install -r requirements.txt 
```

##  向量（embedding）模型
- 支持微调任意开源的embedding模型（bge,m3e等等）

- 支持对两种数据进行微调：
    -  query和正例（负例采用batch内随机负例），
    -  query和正例以及难负例。（负例为对应的难负例，以及batch内随机负例）

微调embedding模型流程,详细的流程可参考模型目录下的Tutorial。
```bash
cd ./rag_retrieval/train/embedding
bash train_embedding.sh
```

## 迟交互式（colbert）模型

- 支持微调开源的bge-m3e模型中的colbert。
- 支持query和正例以及难负例。（负例为对应的难负例，以及batch内随机负例）

微调colbert模型流程，详细的流程可参考模型目录下的Tutorial。
```bash
cd ./rag_retrieval/train/colbert
bash train_colbert.sh
```
## 排序（reranker,cross encoder）模型
- 支持微调任意开源的reranker模型（例如，bge-rerank、bce-rerank等）。
- 支持两种数据进行微调：
    - query和doc的相关性为二分类（1代表相关、0代表不相关）
    - query和doc的相关性为score。（蒸馏任务）

微调reranker模型流程，详细的流程可参考模型目录下的Tutorial。
```bash
cd ./rag_retrieval/train/reranker
bash train_reranker.sh
```


# 实验结果


## reranker模型在 MTEB Reranking 任务的结果


|      **Model**       |  **Model Size(GB)**  |**T2Reranking** | **MMarcoReranking** | **CMedQAv1** | **CMedQAv2** | **Avg** |
|:-----------:|:----------:|:----------:|:-------------:|:--------------:|:---------------:| :---------------:|
|   bge-reranker-base   |  1.11 | 67.28    |      35.46     |      81.27      |       84.10      | 67.03
| bce-reranker-base_v1 |   1.11 |70.25    |      34.13     |      79.64      |       81.31      | 66.33
| rag-retrieval-reranker |  0.41 | 67.33    |      31.57     |      83.54     |       86.03     | 67.12

其中，rag-retrieval-reranker是我们使用RAG-Retrieval代码在hfl/chinese-roberta-wwm-ext模型上训练所得，训练数据使用bge-rerank模型的训练数据.

## colbert模型在 MTEB Reranking 任务的结果

|      **Model**  | **Model Size(GB)**  | **Dim**  | **T2Reranking** | **MMarcoReranking** | **CMedQAv1** | **CMedQAv2** | **Avg** |
|:-----------: |:----------:|:----------:|:----------:|:-------------:|:--------------:|:---------------:| :---------------:|
|   bge-m3-colbert   | 2.24 | 1024 | 66.82 | 26.71    |      75.88     |      76.83      |      61.56      
| rag-retrieval-colbert | 0.41 |  1024|  66.85    |      31.46     |      81.05     |       84.22     | 65.90

其中，rag-retrieval-colbert是我们使用RAG-Retrieval代码在hfl/chinese-roberta-wwm-ext模型上训练所得，训练数据使用bge-rerank模型的训练数据.


## 用领域内数据微调开源的BGE系列模型

|      **Model**  | **T2ranking**  | |
|:-----------: |:----------:|:----------:|
|   bge-v1.5-embedding   | 66.49|  | 
|   bge-v1.5-embedding **finetune**    | 67.15 | **+0.66** | 
|   bge-m3-colbert   | 66.82|  | 
|   bge-m3-colbert **finetune**    | 67.22 | **+0.40** | 
|   bge-reranker-base   | 67.28|  | 
|   bge-reranker-base  **finetune**    | 67.57 | **+0.29** | 

后面带有finetune的代表我们使用RAG-Retrieval在对应开源模型的基础上继续微调所得，训练数据使用T2-Reranking的训练集。

值得注意的是bge的三种开源模型，训练集中已经包含了T2-Reranking，并且该数据较为通用，因此使用该数据继续微调的性能提升效果不大，但是如果使用垂直领域的数据集继续微调开源模型，性能提升会更大。

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NLPJCL/RAG-Retrieval&type=Date)](https://star-history.com/#NLPJCL/RAG-Retrieval&Date)

# License
RAG-Retrieval is licensed under the [MIT License](https://github.com/NLPJCL/RAG-Retrieval/blob/master/LICENSE). 
