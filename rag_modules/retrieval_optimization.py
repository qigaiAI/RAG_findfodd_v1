"""
检索优化模块
"""

import logging
from typing import List, Dict, Any

from config import RAGConfig
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(self, vectorstore: FAISS, chunks: List[Document], stop_documents: List[Document], data_module):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
            stop_documents: 去除停用词后的文档
            data_module: 数据准备模块
        """
        self.config = RAGConfig()
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.stop_documents = stop_documents
        self.data_module = data_module
        self.setup_retrievers()


    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}#设置返回的相似文档数量
        )
        # 打印检索器的类路径（验证归属）
        print(type(self.vector_retriever))
        # 输出示例：<class 'langchain_core.vectorstores.VectorStoreRetriever'>

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.stop_documents,
            k=8
        )



        logger.info("检索器设置完成")
    
    def hybrid_search(self, query: str, top_k: int = 8) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vector_retriever.invoke(query)
        # 直接使用原始查询文本
        bm25_docs_stop = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs_stop)
        return reranked_docs[:top_k]

    def hybrid_search_from_filters(self, query: str, filters_chunks: List[Document], top_k: int = 8) -> List[Document]:
        """
        混合检索 - 从过滤后的文档块中进行相似度检索

        Args:
            query: 查询文本
            filters_chunks: 过滤后的文档块列表
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        if not filters_chunks:
            raise ValueError("过滤后的文档块列表不能为空")

        from .index_construction import IndexConstructionModule
        temp_index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.temp_index_save_path
        )

        # 创建临时向量存储
        temp_vectorstore = FAISS.from_documents(
            documents=filters_chunks,
            embedding=temp_index_module.embeddings
        )
        temp_vector_retriever = temp_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

        # BM25 检索器

        stop_chunks=self.data_module.filter_stop_documents(filters_chunks)#filters_chunks(metadata过滤后的chunks)去掉停用词
        temp_bm25_retriever = BM25Retriever.from_documents(
            stop_chunks,
            k=top_k
        )

        # 分别获取向量检索和BM25检索结果
        vector_docs = temp_vector_retriever.invoke(query)
        query_bm25 = self.data_module.precise_cut(query)
        query_bm25 = self.data_module.filter_stop_words(query_bm25)
        query_bm25 = "".join(query_bm25)
        bm25_docs = temp_bm25_retriever.invoke(query_bm25)


        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]



    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索
        
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量
            
        Returns:
            过滤后的文档列表
        """
        # 先进行混合检索，获取更多候选
        #docs = self.hybrid_search(query, top_k * 5)

        # 应用元数据过滤
        filtered_docs = []
        for doc in self.chunks:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):  #如果过滤条件是列表，检查文档元数据是否在列表中
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
        
        return filtered_docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的哈希作为唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)#.items获取doc_scores的键值对，lambda:x临时定义函数，按第二个元素排序，reverse升序

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs


