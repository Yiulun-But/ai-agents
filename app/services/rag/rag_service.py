"""
Main RAG Service for ai-agents-fork
Orchestrates memory, search, and generation components
"""
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import openai
from ..memory.memory_manager import MemoryManager
from ..search.web_search_service import WebSearchService
from ...core.config import settings

@dataclass
class RAGResponse:
    """Response from RAG system"""
    response: str
    sources: List[str]
    memory_used: int
    web_results: int
    facts_retrieved: int
    processing_time: float
    metadata: Dict[str, Any]
    conversation_id: Optional[str] = None
    error: Optional[str] = None

class RAGService:
    """Main RAG orchestration service for ai-agents-fork"""
    
    def __init__(self):
        """Initialize RAG service with memory and search components"""
        try:
            # Initialize components
            self.memory_manager = MemoryManager("graph")
            self.search_service = WebSearchService()
            
            # Set up OpenAI client
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.LLM_MODEL
            
            print("[RAG_SERVICE] Initialized with graph memory and web search")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize RAG service: {e}")
            raise
    
    async def chat_async(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None,
        digital_human_id: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> RAGResponse:
        """Process a chat message asynchronously with RAG enhancement"""
        
        start_time = time.time()
        
        try:
            # Step 1: Retrieve context using memory system
            print(f"[RAG_SERVICE] Retrieving context for: {user_message[:50]}...")
            retrieval_result = await self.memory_manager.retrieve_context_async(
                user_message, 
                max_results=5,
                user_id=user_id,
                digital_human_id=digital_human_id
            )
            
            # Step 2: Generate response using OpenAI with context
            response_text = await self._generate_response_async(
                user_message, 
                retrieval_result,
                system_prompt
            )
            
            # Step 3: Store the conversation using memory system
            await self.memory_manager.store_conversation_async(
                user_message,
                response_text,
                conversation_id,
                user_id,
                digital_human_id
            )
            
            processing_time = time.time() - start_time
            
            return RAGResponse(
                response=response_text,
                sources=retrieval_result.sources,
                memory_used=len(retrieval_result.memories),
                web_results=len(retrieval_result.web_results),
                facts_retrieved=len(retrieval_result.facts),
                processing_time=round(processing_time, 2),
                metadata=retrieval_result.metadata,
                conversation_id=conversation_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"RAG processing failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            
            return RAGResponse(
                response="I apologize, but I encountered an error processing your message. Please try again.",
                sources=[],
                memory_used=0,
                web_results=0,
                facts_retrieved=0,
                processing_time=processing_time,
                metadata={},
                conversation_id=conversation_id,
                error=error_msg
            )
    
    def chat_sync(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None,
        digital_human_id: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> RAGResponse:
        """Process a chat message synchronously with RAG enhancement"""
        
        start_time = time.time()
        
        try:
            # Step 1: Retrieve context using memory system
            print(f"[RAG_SERVICE] Retrieving context for: {user_message[:50]}...")
            retrieval_result = self.memory_manager.retrieve_context_sync(
                user_message, 
                max_results=5,
                user_id=user_id,
                digital_human_id=digital_human_id
            )
            
            # Step 2: Generate response using OpenAI with context
            response_text = self._generate_response_sync(
                user_message, 
                retrieval_result,
                system_prompt
            )
            
            # Step 3: Store the conversation using memory system
            self.memory_manager.store_conversation_sync(
                user_message,
                response_text,
                conversation_id,
                user_id,
                digital_human_id
            )
            
            processing_time = time.time() - start_time
            
            return RAGResponse(
                response=response_text,
                sources=retrieval_result.sources,
                memory_used=len(retrieval_result.memories),
                web_results=len(retrieval_result.web_results),
                facts_retrieved=len(retrieval_result.facts),
                processing_time=round(processing_time, 2),
                metadata=retrieval_result.metadata,
                conversation_id=conversation_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"RAG processing failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            
            return RAGResponse(
                response="I apologize, but I encountered an error processing your message. Please try again.",
                sources=[],
                memory_used=0,
                web_results=0,
                facts_retrieved=0,
                processing_time=processing_time,
                metadata={},
                conversation_id=conversation_id,
                error=error_msg
            )
    
    async def _generate_response_async(
        self, 
        user_message: str, 
        retrieval_result,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response asynchronously using OpenAI with retrieved context"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_response_sync,
            user_message,
            retrieval_result,
            system_prompt
        )
    
    def _generate_response_sync(
        self, 
        user_message: str, 
        retrieval_result,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using OpenAI with retrieved context"""
        
        # Build context from memory and web search
        context_parts = []
        
        # Add memory context
        if retrieval_result.memories:
            context_parts.append("Previous conversation context:")
            for i, memory in enumerate(retrieval_result.memories[:3], 1):
                context_parts.append(f"{i}. {memory}")
        
        # Add facts
        if retrieval_result.facts:
            context_parts.append("\nKnown facts about the user:")
            for key, value in retrieval_result.facts.items():
                if not key.startswith('_'):  # Skip internal metadata
                    context_parts.append(f"- {key}: {value}")
        
        # Add web search results
        if retrieval_result.web_results:
            context_parts.append("\nRecent information from web search:")
            for i, result in enumerate(retrieval_result.web_results[:3], 1):
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No description')
                context_parts.append(f"{i}. {title}: {snippet}")
        
        # Build the prompt
        default_system_prompt = """You are a helpful AI assistant with access to conversation memory and current web information.

Use the provided context to give accurate, helpful responses. If you use information from web search results, mention that you found recent information. If you reference previous conversations, acknowledge the context naturally.

Be conversational, helpful, and accurate. Don't mention technical details about your memory system or processing unless specifically asked."""
        
        final_system_prompt = system_prompt or default_system_prompt
        
        context_text = "\n".join(context_parts) if context_parts else "No additional context available."
        
        user_prompt = f"""Context:\n{context_text}\n\nUser message: {user_message}

Please respond naturally and helpfully based on the context provided."""
        
        try:
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {e}")
            
            # Fallback response
            if retrieval_result.memories:
                return f"Based on our previous conversations, I understand you're asking about: {user_message}. However, I'm having trouble accessing my language model right now. Could you please try again?"
            else:
                return f"I understand you're asking about: {user_message}. I'm experiencing some technical difficulties with my response generation. Please try again in a moment."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics"""
        try:
            memory_stats = self.memory_manager.get_memory_stats()
            search_stats = self.search_service.get_search_stats()
            
            return {
                'service_name': 'RAG Service',
                'model': self.model,
                'memory_system': memory_stats,
                'search_service': search_stats,
                'capabilities': [
                    'Natural conversation',
                    'Memory of past conversations',
                    'Fact extraction and storage',
                    'Web search for current information',
                    'LangGraph-inspired processing',
                    'Intent classification',
                    'Conditional tool routing',
                    'GraphRAG ready'
                ]
            }
        except Exception as e:
            return {
                'error': str(e),
                'service_name': 'RAG Service'
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on RAG service"""
        try:
            # Check components
            memory_health = self.memory_manager.health_check()
            search_health = self.search_service.health_check()
            
            # Test OpenAI connection
            openai_healthy = True
            try:
                test_response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                openai_healthy = True
            except:
                openai_healthy = False
            
            overall_status = 'healthy'
            if memory_health['status'] != 'healthy' or search_health['status'] == 'unhealthy' or not openai_healthy:
                overall_status = 'degraded'
            if not openai_healthy and search_health['status'] == 'unhealthy':
                overall_status = 'unhealthy'
            
            return {
                'status': overall_status,
                'components': {
                    'memory_manager': memory_health,
                    'search_service': search_health,
                    'openai_client': openai_healthy
                },
                'model': self.model,
                'capabilities_available': overall_status != 'unhealthy'
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def clear_memory_async(self, user_id: Optional[int] = None) -> bool:
        """Clear memory asynchronously"""
        return await self.memory_manager.clear_memory_async(user_id)
    
    def clear_memory_sync(self, user_id: Optional[int] = None) -> bool:
        """Clear memory synchronously"""
        try:
            self.memory_manager.memory_system.clear()
            return True
        except Exception as e:
            print(f"[ERROR] Clear memory failed: {e}")
            return False