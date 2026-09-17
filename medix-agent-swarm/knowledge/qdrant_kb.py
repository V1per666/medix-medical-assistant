"""
医学知识库（Qdrant）

功能：
1. 文档向量化和存储
2. 语义检索
3. 知识库管理

后端：Qdrant 向量数据库（Docker 部署，默认 localhost:6333）
接口与旧版 Milvus 实现保持一致（MedicalKnowledgeBase 单例）
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


class MedicalKnowledgeBase:
    """医学知识库（Qdrant 后端）"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        collection_name: str = "medical_knowledge",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        db_path: Optional[str] = None
    ):
        """
        初始化医学知识库

        Args:
            collection_name: Collection 名称
            embedding_model: Embedding 模型名称或本地路径
            qdrant_host: Qdrant 服务地址
            qdrant_port: Qdrant 服务端口
            db_path: 兼容旧版 Milvus 参数（已废弃，保留签名兼容）
        """
        # 防止重复初始化
        if hasattr(self, '_initialized'):
            return

        self.collection_name = collection_name

        # 初始化 Embedding 模型（支持本地缓存路径）
        local_model_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots"

        if local_model_path.exists():
            # 找到最新的 snapshot
            snapshots = sorted(local_model_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshots:
                model_path = str(snapshots[0])
                logger.info(f"Loading embedding model from local cache: {model_path}")
                self.embedding_model = SentenceTransformer(model_path, device='cpu')
            else:
                logger.info(f"Loading embedding model: {embedding_model}")
                self.embedding_model = SentenceTransformer(embedding_model, device='cpu')
        else:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model, device='cpu')

        self.embedding_dim = self.embedding_model.get_embedding_dimension()
        logger.info(f"Embedding model loaded (dimension={self.embedding_dim})")

        # 初始化 Qdrant 客户端
        logger.info(f"Connecting to Qdrant: {qdrant_host}:{qdrant_port}")
        self.qdrant_client = QdrantClient(url=f"http://{qdrant_host}:{qdrant_port}", timeout=30)
        self._init_collection()

        self._initialized = True

    def _init_collection(self):
        """创建 collection（如果不存在）"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE  # 余弦相似度
                )
            )
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

    @staticmethod
    def _stable_point_id(doc_id: str) -> int:
        """将字符串 doc_id 映射为稳定的 uint64 点 ID（保证重复导入幂等覆盖）"""
        return int(hashlib.md5(doc_id.encode("utf-8")).hexdigest()[:16], 16)

    def _chunk_text(self, text: str, chunk_size: int = 1024, overlap: int = 100) -> List[str]:
        """
        分块文本

        Args:
            text: 原始文本
            chunk_size: 块大小（字符数）
            overlap: 重叠字符数

        Returns:
            文本块列表
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap  # 重叠

        return chunks

    def add_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 1024) -> int:
        """
        添加文档到知识库（支持分块）

        Args:
            documents: 文档列表，每个文档包含 id, content, metadata
            chunk_size: 分块大小（字符数），默认 1024

        Returns:
            成功添加的文档块数量
        """
        if not documents:
            logger.warning("No documents to add")
            return 0

        logger.info(f"Adding {len(documents)} documents to knowledge base (chunk_size={chunk_size})...")

        # 分块
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc["content"], chunk_size=chunk_size)
            for i, chunk in enumerate(chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata["doc_id"] = doc["id"]
                metadata["chunk_id"] = i
                metadata["total_chunks"] = len(chunks)

                all_chunks.append({
                    "doc_id": doc["id"],
                    "content": chunk,
                    "metadata": metadata
                })

        logger.info(f"Split into {len(all_chunks)} chunks")

        # 向量化
        contents = [chunk["content"] for chunk in all_chunks]
        vectors = self.embedding_model.encode(contents, show_progress_bar=True)

        # 构造 Qdrant 点（幂等 upsert）
        points = []
        for i, chunk in enumerate(all_chunks):
            point_id = self._stable_point_id(f"{chunk['doc_id']}#{chunk['metadata']['chunk_id']}")
            points.append(PointStruct(
                id=point_id,
                vector=vectors[i].tolist(),
                payload={
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                }
            ))

        self.qdrant_client.upsert(self.collection_name, points)
        logger.info(f"Successfully added {len(points)} chunks")
        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回top K个结果
            filter_type: 可选的类型过滤（如 "lifestyle", "disease_classification"）

        Returns:
            文档列表，每个文档包含 id, content, metadata, score
        """
        logger.debug(f"Searching for: {query} (top_k={top_k}, filter_type={filter_type})")

        # 向量化查询
        query_vector = self.embedding_model.encode([query])[0]

        # 构建过滤条件
        query_filter = None
        if filter_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.type",
                        match=MatchValue(value=filter_type)
                    )
                ]
            )

        # 检索（qdrant-client >= 1.16 使用 query_points，search 已移除）
        try:
            response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                query_filter=query_filter,
            )
            results = response.points
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        # 格式化结果
        documents = []
        for hit in results:
            try:
                documents.append({
                    "id": hit.id,
                    "content": hit.payload.get("content", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "score": hit.score  # Qdrant COSINE 直接返回余弦相似度
                })
            except Exception as e:
                logger.warning(f"Failed to parse result: {e}")
                continue

        logger.debug(f"Found {len(documents)} documents")
        return documents

    def delete_collection(self):
        """删除 collection（用于测试）"""
        collections = self.qdrant_client.get_collections().collections
        if self.collection_name in [c.name for c in collections]:
            self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")

    def count_documents(self) -> int:
        """统计文档数量"""
        try:
            return self.qdrant_client.count(self.collection_name).count
        except Exception as e:
            logger.warning(f"Failed to count documents: {e}")
            return 0
