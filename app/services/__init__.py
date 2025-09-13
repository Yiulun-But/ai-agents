"""
Services module for ai-agents-fork
Enhanced with RAG capabilities as the default conversation pipeline
"""

# RAG Services (Default Pipeline)
from .rag import RAGService, ConversationService as RAGConversationService
from .memory import MemoryManager, GraphMemorySystem
from .search import WebSearchService

# Legacy services for compatibility
from .conversation_service import ConversationService
from .langgraph_service import LangGraphService

__all__ = [
    # RAG Pipeline (Primary)
    "RAGService",
    "RAGConversationService", 
    "MemoryManager",
    "GraphMemorySystem",
    "WebSearchService",
    
    # Legacy/Compatibility
    "ConversationService",
    "LangGraphService"
]

# Service initialization info
PIPELINE_INFO = {
    "default_pipeline": "RAG",
    "features": [
        "Memory-enhanced conversations",
        "Real-time web search integration", 
        "Semantic context retrieval",
        "Fact extraction and persistence",
        "Intent-based routing",
        "GraphRAG ready architecture"
    ],
    "components": {
        "memory": "ChromaDB + OpenAI Embeddings",
        "search": "Multi-provider (DuckDuckGo, Serper, Mock)",
        "llm": "OpenAI GPT models",
        "routing": "LangGraph-inspired conditional flows"
    }
}