"""
Enhanced Conversation Service for ai-agents-fork
Integrates RAG capabilities with the existing conversation system
"""
from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio
import json
import uuid
from datetime import datetime

from .rag_service import RAGService, RAGResponse
from ...core.config import settings

class ConversationService:
    """Enhanced conversation service with RAG capabilities"""
    
    def __init__(self):
        """Initialize conversation service with RAG"""
        self.rag_service = RAGService()
        print("[CONVERSATION_SERVICE] Initialized with RAG capabilities")
    
    async def process_message_async(
        self,
        message: str,
        user_id: Optional[int] = None,
        digital_human_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Process a conversation message with RAG enhancement"""
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        try:
            if stream:
                # For streaming, we need to handle it differently
                return await self._process_streaming_message(
                    message, user_id, digital_human_id, conversation_id, system_prompt
                )
            else:
                # Regular non-streaming response
                rag_response = await self.rag_service.chat_async(
                    user_message=message,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    digital_human_id=digital_human_id,
                    system_prompt=system_prompt
                )
                
                return self._format_response(rag_response, user_id, digital_human_id)
                
        except Exception as e:
            print(f"[CONVERSATION_SERVICE] Error processing message: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process conversation message",
                "conversation_id": conversation_id
            }
    
    async def _process_streaming_message(
        self,
        message: str,
        user_id: Optional[int],
        digital_human_id: Optional[int],
        conversation_id: str,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Process message for streaming response"""
        
        # For now, we'll simulate streaming by getting the full response first
        # In a real implementation, you'd want to stream from the OpenAI API
        rag_response = await self.rag_service.chat_async(
            user_message=message,
            conversation_id=conversation_id,
            user_id=user_id,
            digital_human_id=digital_human_id,
            system_prompt=system_prompt
        )
        
        # Return the full response with streaming indicators
        response_data = self._format_response(rag_response, user_id, digital_human_id)
        response_data["streaming"] = True
        response_data["stream_complete"] = True
        
        return response_data
    
    async def stream_response_async(
        self,
        message: str,
        user_id: Optional[int] = None,
        digital_human_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response using Server-Sent Events format"""
        
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        try:
            # Get RAG response
            rag_response = await self.rag_service.chat_async(
                user_message=message,
                conversation_id=conversation_id,
                user_id=user_id,
                digital_human_id=digital_human_id,
                system_prompt=system_prompt
            )
            
            # Simulate streaming by yielding chunks
            response_text = rag_response.response
            chunk_size = 20  # Characters per chunk
            
            # Yield metadata first
            metadata = {
                "type": "metadata",
                "conversation_id": conversation_id,
                "sources": rag_response.sources,
                "memory_used": rag_response.memory_used,
                "web_results": rag_response.web_results,
                "processing_time": rag_response.processing_time
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # Yield response chunks
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                chunk_data = {
                    "type": "chunk",
                    "content": chunk,
                    "chunk_id": i // chunk_size
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.05)  # Small delay to simulate streaming
            
            # Yield completion signal
            completion_data = {
                "type": "complete",
                "conversation_id": conversation_id,
                "total_chunks": (len(response_text) + chunk_size - 1) // chunk_size
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "error": str(e),
                "conversation_id": conversation_id
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    def _format_response(
        self, 
        rag_response: RAGResponse, 
        user_id: Optional[int], 
        digital_human_id: Optional[int]
    ) -> Dict[str, Any]:
        """Format RAG response for API consumption"""
        
        return {
            "success": True,
            "data": {
                "message": rag_response.response,
                "conversation_id": rag_response.conversation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "sources": rag_response.sources,
                    "memory_used": rag_response.memory_used,
                    "web_results": rag_response.web_results,
                    "facts_retrieved": rag_response.facts_retrieved,
                    "processing_time": rag_response.processing_time,
                    "rag_metadata": rag_response.metadata,
                    "user_id": user_id,
                    "digital_human_id": digital_human_id
                }
            },
            "error": rag_response.error
        }
    
    async def get_conversation_history_async(
        self,
        conversation_id: str,
        user_id: Optional[int] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get conversation history (placeholder for future implementation)"""
        
        # This would integrate with your existing database to fetch conversation history
        # For now, return a placeholder response
        
        return {
            "success": True,
            "data": {
                "conversation_id": conversation_id,
                "messages": [],  # Would be populated from database
                "total_messages": 0,
                "limit": limit
            }
        }
    
    async def delete_conversation_async(
        self,
        conversation_id: str,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Delete a conversation (placeholder for future implementation)"""
        
        # This would integrate with your existing database to delete conversations
        # For now, return a placeholder response
        
        return {
            "success": True,
            "data": {
                "conversation_id": conversation_id,
                "deleted": True
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation service statistics"""
        try:
            rag_stats = self.rag_service.get_stats()
            
            return {
                "service_name": "Enhanced Conversation Service",
                "rag_pipeline": True,
                "features": [
                    "RAG-enhanced responses",
                    "Memory persistence",
                    "Web search integration",
                    "Streaming support",
                    "Context awareness",
                    "Multi-user support",
                    "Digital human integration"
                ],
                "rag_service": rag_stats
            }
        except Exception as e:
            return {
                "service_name": "Enhanced Conversation Service",
                "error": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on conversation service"""
        try:
            rag_health = self.rag_service.health_check()
            
            return {
                "status": rag_health["status"],
                "service_name": "Enhanced Conversation Service",
                "rag_service": rag_health,
                "features_available": rag_health["capabilities_available"]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service_name": "Enhanced Conversation Service",
                "error": str(e)
            }
    
    async def clear_memory_async(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Clear conversation memory"""
        try:
            success = await self.rag_service.clear_memory_async(user_id)
            return {
                "success": success,
                "message": "Memory cleared successfully" if success else "Failed to clear memory",
                "user_id": user_id
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error clearing memory: {str(e)}",
                "user_id": user_id
            }