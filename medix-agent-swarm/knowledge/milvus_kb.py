"""
兼容层：医学知识库

后端已从 Milvus Lite 迁移至 Qdrant，本模块保留文件名以兼容既有 import。
实际实现见 knowledge/qdrant_kb.py
"""
from knowledge.qdrant_kb import MedicalKnowledgeBase

__all__ = ["MedicalKnowledgeBase"]
