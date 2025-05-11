import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os.path
import time
import hashlib
import numpy as np
from typing import Literal
import tqdm
import voyageai
from sentence_transformers import SentenceTransformer, CrossEncoder
import commercial_embedding_api
import faiss
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import bm25s
import Stemmer
import pickle
from FlagEmbedding import FlagReranker, FlagLLMReranker



def find_topk_via_faiss(source_vecs: np.ndarray, target_vecs: np.ndarray, topk: int):
    faiss_index = faiss.IndexFlatIP(target_vecs.shape[1])
    faiss_index.add(target_vecs)
    res_distance, res_index = faiss_index.search(source_vecs, topk)
    return res_index, res_distance


def find_topk_by_bm25(
    query_list: list[str],
    passage_list: list[str],
    topk: int,
):
    stemmer = Stemmer.Stemmer("english")
    # Tokenize the corpus and only keep the ids (faster and saves memory)
    corpus_tokens = bm25s.tokenize(passage_list, stopwords="en", stemmer=stemmer)
    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    # Query the corpus
    query_tokens = bm25s.tokenize(query_list, stemmer=stemmer)
    # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
    # To return docs instead of IDs, set the `corpus=corpus` parameter.
    res_index, res_distance = retriever.retrieve(query_tokens, k=topk)
    
    return res_index, res_distance

def find_topk_by_single_vecs(
    embedding_model_name_or_path: str,
    model_type: Literal["local", "api"],
    query_list: list[str],
    passage_list: list[str],
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find topk passages for each query using single vector.
    Args:
        embedding_model_name_or_path: model name or path
        model_type: model type, either 'local' or 'api'
        query_list: list of query strings
        passage_list: list of passage strings
        topk: number of top passages to return
    Returns:
        topk_index: numpy array of shape (len(query_list), topk)
        topk_scores: numpy array of shape (len(query_list), topk)
    """
    if model_type == "local":
        model = SentenceTransformer(
            embedding_model_name_or_path, trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.float16,
            }
        )
        print(f"Model loaded from {embedding_model_name_or_path}")
        print(f"Model dtype: {next(model.named_parameters())[1].dtype}")
    elif model_type == "api":
        pass
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if "jina-embeddings-v3" in embedding_model_name_or_path:
        model.max_seq_length = 8192     
        q_vecs = model.encode(
            query_list,
            task="retrieval.query",
            prompt_name="retrieval.query",
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True,
        )
        p_vecs = model.encode(
            passage_list,
            task="retrieval.passage",
            prompt_name="retrieval.passage",
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
    elif "bge-m3" in embedding_model_name_or_path or "jina-embeddings-v2-base-en" in embedding_model_name_or_path:
        model.max_seq_length = 8192
        
        pool = model.start_multi_process_pool() 
        q_vecs = model.encode_multi_process(
            query_list,
            pool,
            show_progress_bar=True,
            batch_size=32,
            chunk_size=512,
            normalize_embeddings=True,
        )
        p_vecs = model.encode_multi_process(
            passage_list,
            pool,
            show_progress_bar=True,
            batch_size=32,
            chunk_size=512,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    elif "jasper_en_vision_language_v1" in embedding_model_name_or_path:
        model.max_seq_length = 8192
        pool = model.start_multi_process_pool() 
        q_vecs = model.encode_multi_process(
            query_list,
            pool,
            show_progress_bar=True,
            prompt_name="s2p_query",
            batch_size=32,
            chunk_size=512,
            normalize_embeddings=True,
        )
        p_vecs = model.encode_multi_process(
            passage_list,
            pool,
            show_progress_bar=True,
            batch_size=8,
            chunk_size=512,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    elif "stella_en_400M_v5" in embedding_model_name_or_path:
        model.max_seq_length = 8192
        pool = model.start_multi_process_pool() 
        q_vecs = model.encode_multi_process(
            query_list,
            pool,
            show_progress_bar=True,
            prompt_name="s2p_query",
            batch_size=32,
            chunk_size=512,
            normalize_embeddings=True,
        )
        p_vecs = model.encode_multi_process(
            passage_list,
            pool,
            show_progress_bar=True,
            batch_size=8,
            chunk_size=512,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    elif "nvidia" in embedding_model_name_or_path:
        task_name_to_instruct = {
            "example": "Given a question, retrieve passages that answer the question",
        }
        query_prefix = "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "

        def add_eos(input_examples):
            input_examples = [
                input_example + model.tokenizer.eos_token
                for input_example in input_examples
            ]
            return input_examples

        model.max_seq_length = 8192
        # model.max_seq_length = 32768
        model.tokenizer.padding_side = "right"
        
        pool = model.start_multi_process_pool() 
        q_vecs = model.encode_multi_process(
            add_eos(query_list),
            pool,
            show_progress_bar=True,
            batch_size=2,
            chunk_size=256,
            normalize_embeddings=True,
            prompt=query_prefix,
        )
        p_vecs = model.encode_multi_process(
            add_eos(passage_list),
            pool,
            show_progress_bar=True,
            batch_size=2,
            chunk_size=256,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    elif "gte-Qwen2" in embedding_model_name_or_path:
        model.max_seq_length = 8192
        pool = model.start_multi_process_pool() 
        q_vecs = model.encode_multi_process(
            query_list,
            pool,
            show_progress_bar=True,
            batch_size=2,
            chunk_size=256,
            normalize_embeddings=True,
            prompt_name="query",
        )
        p_vecs = model.encode_multi_process(
            passage_list,
            pool,
            show_progress_bar=True,
            batch_size=2,
            chunk_size=256,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    elif model_type == "api":
        if "openai" in embedding_model_name_or_path:
            encoder = commercial_embedding_api.OpenAIEncoder()
            q_vecs = encoder.encode(
                sentences=query_list,
                normalize_embeddings=True,
            )
            p_vecs = encoder.encode(
                sentences=passage_list,
                normalize_embeddings=True,
            )
        elif "cohere" in embedding_model_name_or_path:
            encoder = commercial_embedding_api.CohereEncoder()
            q_vecs = encoder.encode(
                sentences=query_list,
                normalize_embeddings=True,
                prompt_name="search_query",
            )
            p_vecs = encoder.encode(
                sentences=passage_list,
                normalize_embeddings=True,
                prompt_name="search_document",
            )
        elif "voyage" in embedding_model_name_or_path:
            encoder = commercial_embedding_api.VoyageEncoder()
            q_vecs = encoder.encode(
                sentences=query_list,
                normalize_embeddings=True,
                prompt_name="query",
                output_dimension=2048,
            )
            p_vecs = encoder.encode(
                sentences=passage_list,
                normalize_embeddings=True,
                prompt_name="document",
                output_dimension=2048,
            )

        elif "jina" in embedding_model_name_or_path:
            encoder = commercial_embedding_api.JinaEncoder()
            q_vecs = encoder.encode(
                sentences=query_list,
                normalize_embeddings=True,
                prompt_name="retrieval.query",
                output_dimension=1024,
            )
            p_vecs = encoder.encode(
                sentences=passage_list,
                normalize_embeddings=True,
                prompt_name="retrieval.passage",
                output_dimension=1024,
            )
        else:
            raise Exception(f"Unsupported api model: {embedding_model_name_or_path}")
    else:
        raise Exception(f"Unsupported model: {embedding_model_name_or_path}")
    
    # search topk
    topk_index, topk_scores = find_topk_via_faiss(q_vecs, p_vecs, topk)
    
    return topk_index, topk_scores

def find_topk_by_multi_vecs(
    embedding_model_name_or_path: str,
    model_type: Literal["local", "api"],
    query_list: list[str],
    passage_list: list[str],
    topk: int,
) -> np.ndarray:
    """
    Find topk passages for each query using multiple vectors.
    Args:
        embedding_model_name_or_path: model name or path
        model_type: model type, either 'local' or 'api'
        query_list: list of query strings
        passage_list: list of passage strings
        topk: number of top passages to return
    Returns:
        topk_index: numpy array of shape (len(query_list), topk)
        topk_scores: numpy array of shape (len(query_list), topk)
    """
    pass

def find_topk_by_reranker(
    reranker_model_name_or_path: str,
    embedding_model_name_or_path: str,
    model_type: Literal["local", "api"],
    query_list: list[str],
    passage_list: list[str],
    topk: int,
    cache_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find topk passages for each query using reranker.
    Args:
        reranker_model_name_or_path: model name or path
        embedding_model_name_or_path: model name or path for first stage retrieval
        model_type: model type, either 'local' or 'api'
        query_list: list of query strings
        passage_list: list of passage strings
        topk: number of top passages to return
        cache_path: path to cache the first stage retrieval results and reranker scores
    Returns:
        topk_index: numpy array of shape (len(query_list), topk)
        topk_scores: numpy array of shape (len(query_list), topk)
    """
    pass

def find_topk_by_reranker(
    reranker_model_name_or_path: str,
    embedding_model_name_or_path: str,
    reranker_model_type: Literal["local", "api"],
    embedding_model_type: Literal["local", "api"],
    query_list: list[str],
    passage_list: list[str],
    topk: int,
    recall_topk: int = 100,
    cache_path: str = "./rerank_cache.pickle",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find topk passages for each query using reranker.
    支持的模型：
    local: bge-reranker-v2-m3; jina-reranker-v2-base-multilingual; gte-multilingual-reranker-base
    api: rerank-2
    Args:
        reranker_model_name_or_path: model name or path
        embedding_model_name_or_path: model name or path for first stage retrieval
        reranker_model_type: model type, either 'local' or 'api'
        embedding_model_type: model type, either 'local' or 'api'
        query_list: list of query strings
        passage_list: list of passage strings
        topk: number of top passages to return
        recall_topk: number of top passages in first stage (i.e. recall)
        cache_path: path to cache the first stage retrieval results and reranker scores. If cache path is None, do not use and save cache.
    Returns:
        topk_index: numpy array of shape (len(query_list), topk)
        topk_scores: numpy array of shape (len(query_list), topk)
    """
    # 读取cache
    do_cache = cache_path is not None
    cache_di = {}
    if do_cache:
        recall_key = "Recall+--+{}+--+{}+--+{}+--+{}+--+{}".format(
            embedding_model_name_or_path,
            embedding_model_type,
            "\n".join(sorted(query_list)),
            "\n".join(sorted(passage_list)),
            str(recall_topk)
        )
        recall_key = hashlib.md5(recall_key.encode('utf-8')).hexdigest()
        # TODO 不做rerank的缓存，那不然逻辑太复杂了，需要针对pair对得分进行计算，除非速度慢到无法忍受，否则不做
        rerank_key = "Rerank+--+{}+--+{}+--+{}+--+{}".format(
            reranker_model_name_or_path,
            reranker_model_type,
            recall_key,
            str(topk)
        )
        recall_topk_index_key, recall_topk_scores_key = f"{recall_key}_index", f"{recall_key}_scores"
        if os.path.exists(cache_path):
            print(f"从{cache_path}中加载缓存")
            with open(cache_path, 'rb') as f:
                cache_di = pickle.load(f)
                # print("cache_di", cache_di)
        else:
            print(f"{cache_path}不存在，得到搜索排序结果后，会把结果存储该路径")
    else:
        print("cache_path 为None, 将不执行任何的cache操作！！！！！！！！！")
    # 获取recall topk 结果 并进行相应的缓存操作
    if do_cache and recall_topk_index_key in cache_di and recall_topk_scores_key in cache_di:
        print(f"从缓存加载recall_topk_index和recall_topk_scores")
        topk_index, topk_scores = cache_di[recall_topk_index_key], cache_di[recall_topk_scores_key]
    else:
        print("无缓存或缺少对应的key，从0开始计算recall_topk_index和recall_topk_scores")
        topk_index, topk_scores = find_topk_by_single_vecs(
            embedding_model_name_or_path=embedding_model_name_or_path,
            model_type=embedding_model_type,
            query_list=query_list,
            passage_list=passage_list,
            topk=recall_topk,
        )
        # 对缓存做操作
        if do_cache:
            print("把recall结果存储到缓存中")
            cache_di[recall_topk_index_key] = topk_index
            cache_di[recall_topk_scores_key] = topk_scores
            with open(cache_path, "wb") as f:
                pickle.dump(cache_di, f)

    # 加载rerank 模型 并获取rerank topk 结果 并进行相应的缓存操作
    rerank_topk_index, rerank_topk_scores = [], []
    if reranker_model_type == "local":
        if "bge-reranker-v2-m3" in reranker_model_name_or_path:
            reranker = FlagReranker(reranker_model_name_or_path, use_fp16=True)
            for query, topk_passage_ids in tqdm.tqdm(zip(query_list, topk_index), disable=False, desc=f"rerank..."):
                scores = reranker.compute_score(
                    [(query, passage_list[pid]) for pid in topk_passage_ids], normalize=True, max_length=8192, batch_size=512
                )
                pid_scores = list(zip(topk_passage_ids, scores))
                pid_scores.sort(key=lambda x: x[1], reverse=True)
                rerank_topk_index.append([item[0] for item in pid_scores][:topk])
                rerank_topk_scores.append([item[1] for item in pid_scores][:topk])
        elif "bge-reranker-v2-gemma" in reranker_model_name_or_path:
            reranker = FlagLLMReranker(reranker_model_name_or_path, use_fp16=True)
            for query, topk_passage_ids in tqdm.tqdm(zip(query_list, topk_index), disable=False, desc=f"rerank..."):
                scores = reranker.compute_score(
                    [(query, passage_list[pid]) for pid in topk_passage_ids], normalize=True, max_length=8192, batch_size=4
                )
                pid_scores = list(zip(topk_passage_ids, scores))
                pid_scores.sort(key=lambda x: x[1], reverse=True)
                rerank_topk_index.append([item[0] for item in pid_scores][:topk])
                rerank_topk_scores.append([item[1] for item in pid_scores][:topk])
        elif "jina-reranker-v2-base-multilingual" in reranker_model_name_or_path:
            reranker = CrossEncoder(
                reranker_model_name_or_path,
                automodel_args={"torch_dtype": "auto"},
                trust_remote_code=True,
            )
            reranker.model.cuda().half()
            for query, topk_passage_ids in tqdm.tqdm(zip(query_list, topk_index), disable=False, desc=f"rerank..."):
                scores = reranker.predict(
                    [(query, passage_list[pid]) for pid in topk_passage_ids], convert_to_tensor=True, batch_size=512
                ).tolist()
                pid_scores = list(zip(topk_passage_ids, scores))
                pid_scores.sort(key=lambda x: x[1], reverse=True)
                rerank_topk_index.append([item[0] for item in pid_scores][:topk])
                rerank_topk_scores.append([item[1] for item in pid_scores][:topk])
        elif "gte-multilingual-reranker-base" in reranker_model_name_or_path:
            BSZ = 256
            tokenizer = AutoTokenizer.from_pretrained(reranker_model_name_or_path)
            reranker = AutoModelForSequenceClassification.from_pretrained(
                reranker_model_name_or_path, trust_remote_code=True,
                torch_dtype=torch.float16
            ).cuda()
            reranker.eval()

            with torch.no_grad():
                for query, topk_passage_ids in tqdm.tqdm(zip(query_list, topk_index), disable=False, desc=f"rerank..."):
                    pairs = [(query, passage_list[pid]) for pid in topk_passage_ids]
                    scores = []
                    for start_id in range(0, len(pairs), BSZ):
                        inputs = tokenizer(
                            pairs[start_id:start_id + BSZ],
                            padding=True,
                            truncation=True,
                            return_tensors='pt',
                            max_length=8190,
                        )
                        batch_scores = reranker(
                            **{k: v.cuda() for k, v in inputs.items()}, return_dict=True
                        ).logits.view(-1, ).float().cpu().numpy().tolist()
                        scores.extend(batch_scores)

                    pid_scores = list(zip(topk_passage_ids, scores))
                    pid_scores.sort(key=lambda x: x[1], reverse=True)
                    rerank_topk_index.append([item[0] for item in pid_scores][:topk])
                    rerank_topk_scores.append([item[1] for item in pid_scores][:topk])

    else:
        assert reranker_model_type == "api"
        if reranker_model_name_or_path == "rerank-2":
            client = voyageai.Client(
                api_key=os.environ["VOYAGE_KEY"],
                max_retries=32
            )
            for query, topk_passage_ids in tqdm.tqdm(zip(query_list, topk_index), disable=False, desc=f"rerank..."):
                documents = [passage_list[pid] for pid in topk_passage_ids]
                resp = client.rerank(query=query, documents=documents, model=reranker_model_name_or_path, top_k=topk)
                pid_scores = [[topk_passage_ids[r.index], r.relevance_score] for r in resp.results]
                rerank_topk_index.append([item[0] for item in pid_scores])
                rerank_topk_scores.append([item[1] for item in pid_scores])
                # TODO 根据限制适当调整
                time.sleep(1)

    # 合并结果
    return np.array(rerank_topk_index), np.array(rerank_topk_scores)


if __name__ == "__main__":
    query_list = [
        "What is machine learning?",
        "How does natural language processing work?",
    ]
    passage_list = [
        "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms allowing computers to learn from and make predictions on data. It involves training models on historical data to identify patterns and make informed decisions without explicit programming.",
        "Natural language processing (NLP) enables computers to understand, interpret, and generate human language in valuable and meaningful ways. It combines computational linguistics—rule-based modeling of human language—with statistical, machine learning, and deep learning models.",
        "Neural networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data. They are composed of layers of interconnected nodes (neurons) that process information through a system of inputs and outputs.",
        "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects and then react to what they 'see'.",
        "Supervised learning uses labeled data to train algorithms to map inputs to outputs. Unsupervised learning, in contrast, deals with unlabeled data to find hidden patterns or groupings within the data. Both are fundamental to machine learning but serve different purposes.",
        "A convolutional neural network (CNN) is a type of deep learning model specifically designed for processing grid-like data such as images. CNNs use convolutional layers to automatically extract features from images, reducing the need for manual feature engineering.",
        "The transformer architecture is a deep learning model that relies entirely on self-attention mechanisms to process sequences. Unlike recurrent neural networks, transformers can process entire sequences in parallel, making them more efficient for tasks like machine translation and text generation.",
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for actions taken, aiming to maximize cumulative rewards over time through trial and error.",
        "Recommendation systems are software tools and techniques providing suggestions for items that are most pertinent to a particular user. Key components include user profiling, item representation, similarity computation, and ranking algorithms to deliver personalized content.",
        "Generative adversarial networks (GANs) consist of two neural networks—generators and discriminators—that compete against each other. The generator creates synthetic data, while the discriminator evaluates its authenticity. This adversarial process improves the generator's ability to produce realistic data."
    ]
    
    #################################### BM25 ###############################################
    topk_index, topk_scores = find_topk_by_bm25(
        query_list=query_list,    
        passage_list=passage_list,
        topk=5,
    )
    for query, ids, scores in zip(query_list, topk_index, topk_scores):
        for idx, score in zip(ids, scores):
            print(query, passage_list[idx], score, sep="  <---->  ")
    
    #################################### Single Embedding - Test Local###############################################
    # topk_index, topk_scores = find_topk_by_single_vecs(
    #     embedding_model_name_or_path="/data/zzy/models/BAAI/bge-m3",
    #     model_type="local",
    #     query_list=query_list,    
    #     passage_list=passage_list,
    #     topk=5,
    # )
    # for query, ids, scores in zip(query_list, topk_index, topk_scores):
    #     for idx, score in zip(ids, scores):
    #         print(query, passage_list[idx], score, sep="  <---->  ")
    
    #######################################  ReRanker - Test Local###################################################
    # topk_index, topk_scores = find_topk_by_reranker(
    #     reranker_model_name_or_path="/data/zzy/models/BAAI/bge-reranker-v2-gemma",
    #     embedding_model_name_or_path="/data/zzy/models/BAAI/bge-m3",
    #     reranker_model_type="local",
    #     embedding_model_type="local",
    #     query_list=query_list,
    #     passage_list=passage_list,
    #     topk=5,
    #     recall_topk=9,
    #     cache_path="./rerank_cache.pickle",
    # )
    # for query, ids, scores in zip(query_list, topk_index, topk_scores):
    #     for idx, score in zip(ids, scores):
    #         print(query, passage_list[idx], score, sep="  <---->  ")

    #######################################   ReRanker - Test API ###################################################
    # os.environ["VOYAGE_KEY"] = "xxx"
    # topk_index, topk_scores = find_topk_by_reranker(
    #     reranker_model_name_or_path="rerank-2",
    #     embedding_model_name_or_path="/data/zzy/models/BAAI/bge-m3",
    #     reranker_model_type="api",
    #     embedding_model_type="local",
    #     query_list=query_list,
    #     passage_list=passage_list,
    #     topk=5,
    #     recall_topk=9,
    #     cache_path="./rerank_cache.pickle",
    # )
    # for query, ids, scores in zip(query_list, topk_index, topk_scores):
    #     for idx, score in zip(ids, scores):
    #         print(query, passage_list[idx], score, sep="  <---->  ")
