"""
Memory Manager for RAG system integration with ai-agents-fork
Provides high-level interface for memory operations
"""
from typing import Dict, List, Any, Optional
from .graph_memory import GraphMemorySystem
from .base_memory import RetrievalResult

class MemoryManager:
    """High-level memory management service for ai-agents-fork"""
    
    def __init__(self, memory_type: str = "graph"):
        """Initialize memory manager with specified memory system"""
        self.memory_type = memory_type
        
        if memory_type == "graph":
            self.memory_system = GraphMemorySystem("AI Agents RAG Memory")
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
        
        print(f"[MEMORY_MANAGER] Initialized with {memory_type} memory system")
    
    async def store_conversation_async(
        self,
        user_message: str,
        assistant_response: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None,
        digital_human_id: Optional[int] = None
    ) -> bool:
        """Store conversation asynchronously with ai-agents-fork context"""
        try:
            context = {}
            if conversation_id:
                context['conversation_id'] = conversation_id
            if user_id:
                context['user_id'] = user_id
            if digital_human_id:
                context['digital_human_id'] = digital_human_id
            
            # Execute in thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.memory_system.store_conversation,
                user_message,
                assistant_response,
                context
            )
            
            return True
        except Exception as e:
            print(f"[MEMORY_MANAGER] Store conversation failed: {e}")
            return False
    
    async def retrieve_context_async(
        self,
        query: str,
        max_results: int = 5,
        user_id: Optional[int] = None,
        digital_human_id: Optional[int] = None
    ) -> RetrievalResult:
        """Retrieve conversation context asynchronously"""
        try:
            # Execute in thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.memory_system.retrieve_context,
                query,
                max_results
            )
            
            # Add user/digital_human context to metadata
            if user_id:
                result.metadata['user_id'] = user_id
            if digital_human_id:
                result.metadata['digital_human_id'] = digital_human_id
            
            return result
        except Exception as e:
            print(f"[MEMORY_MANAGER] Retrieve context failed: {e}")
            return RetrievalResult()
    
    def store_conversation_sync(
        self,
        user_message: str,
        assistant_response: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None,
        digital_human_id: Optional[int] = None
    ) -> bool:
        """Store conversation synchronously"""
        try:
            context = {}
            if conversation_id:
                context['conversation_id'] = conversation_id
            if user_id:
                context['user_id'] = user_id
            if digital_human_id:
                context['digital_human_id'] = digital_human_id
            
            self.memory_system.store_conversation(
                user_message,
                assistant_response,
                context
            )
            
            return True
        except Exception as e:
            print(f"[MEMORY_MANAGER] Store conversation failed: {e}")
            return False
    
    def retrieve_context_sync(
        self,
        query: str,
        max_results: int = 5,
        user_id: Optional[int] = None,
        digital_human_id: Optional[int] = None
    ) -> RetrievalResult:
        """Retrieve conversation context synchronously"""
        try:
            result = self.memory_system.retrieve_context(query, max_results)
            
            # Add user/digital_human context to metadata
            if user_id:
                result.metadata['user_id'] = user_id
            if digital_human_id:
                result.metadata['digital_human_id'] = digital_human_id
            
            return result
        except Exception as e:
            print(f"[MEMORY_MANAGER] Retrieve context failed: {e}")
            return RetrievalResult()
    
    async def clear_memory_async(self, user_id: Optional[int] = None) -> bool:
        """Clear memory asynchronously"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.memory_system.clear)
            return True
        except Exception as e:
            print(f"[MEMORY_MANAGER] Clear memory failed: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            stats = self.memory_system.get_stats()
            stats['memory_manager_type'] = self.memory_type
            return stats
        except Exception as e:
            print(f"[MEMORY_MANAGER] Get stats failed: {e}")
            return {
                'error': str(e),
                'memory_manager_type': self.memory_type
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on memory system"""
        try:
            # Test basic functionality
            test_result = self.memory_system.get_stats()
            
            return {
                'status': 'healthy',
                'memory_type': self.memory_type,
                'collection_available': self.memory_system.collection is not None,
                'openai_client_available': self.memory_system.openai_client is not None,
                'chroma_client_available': self.memory_system.chroma_client is not None,
                'conversation_count': test_result.get('conversation_count', 0)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'memory_type': self.memory_type,
                'error': str(e)
            }