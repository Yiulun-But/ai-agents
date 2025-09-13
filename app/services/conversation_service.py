from sqlalchemy.orm import Session
from typing import List, Optional, Tuple, Generator, Dict, Any
from app.repositories.conversation_repository import ConversationRepository, MessageRepository
from app.services.langgraph_service import LangGraphService
from app.services.rag.conversation_service import ConversationService as RAGConversationService
from app.core.models import Conversation, Message, DigitalHuman
from app.schemas.conversation import *
import json
import asyncio


class ConversationService:
    
    def __init__(self, db: Session, langgraph_service: LangGraphService):
        self.db = db
        self.conversation_repo = ConversationRepository(db)
        self.message_repo = MessageRepository(db)
        # Keep LangGraph service for thread ID generation only
        self.langgraph_service = langgraph_service
        
        # Initialize RAG conversation service as the default pipeline
        self.rag_service = RAGConversationService()
        print("[CONVERSATION_SERVICE] RAG pipeline initialized as default")
    
    def create_conversation(
        self,
        conversation_data: ConversationCreate,
        user_id: int
    ) -> ConversationResponse:
        thread_id = self.langgraph_service.create_thread_id()
        
        conversation = self.conversation_repo.create_conversation(
            conversation_data, user_id, thread_id
        )
        
        return ConversationResponse.from_orm(conversation)
    
    def get_conversation_by_id(
        self,
        conversation_id: int,
        user_id: int
    ) -> Optional[ConversationResponse]:
        conversation = self.conversation_repo.get_conversation_by_id(
            conversation_id, user_id
        )
        
        if not conversation:
            return None
        
        return ConversationResponse.from_orm(conversation)
    
    def get_conversations_paginated(
        self,
        page_request: ConversationPageRequest,
        user_id: int
    ) -> Tuple[List[ConversationResponse], int]:
        conversations, total = self.conversation_repo.get_conversations_paginated(
            page_request, user_id
        )
        
        conversation_responses = [
            ConversationResponse.from_orm(conv) for conv in conversations
        ]
        
        return conversation_responses, total
    
    def update_conversation(
        self,
        conversation_id: int,
        conversation_data: ConversationUpdate,
        user_id: int
    ) -> Optional[ConversationResponse]:
        conversation = self.conversation_repo.update_conversation(
            conversation_id, conversation_data, user_id
        )
        
        if not conversation:
            return None
        
        return ConversationResponse.from_orm(conversation)
    
    def delete_conversation(self, conversation_id: int, user_id: int) -> bool:
        return self.conversation_repo.delete_conversation(conversation_id, user_id)
    
    def get_conversation_with_messages(
        self,
        conversation_id: int,
        user_id: int,
        message_limit: Optional[int] = None
    ) -> Optional[ConversationWithMessages]:
        conversation = self.conversation_repo.get_conversation_by_id(
            conversation_id, user_id
        )
        
        if not conversation:
            return None
        
        messages = self.message_repo.get_conversation_messages(
            conversation_id, message_limit
        )
        
        conversation_response = ConversationResponse.from_orm(conversation)
        message_responses = [MessageResponse.from_orm(msg) for msg in messages]
        
        return ConversationWithMessages(
            **conversation_response.dict(),
            messages=message_responses
        )
    
    def send_message(
        self,
        conversation_id: int,
        message_content: str,
        user_id: int
    ) -> Optional[MessageResponse]:
        conversation = self.conversation_repo.get_conversation_by_id(
            conversation_id, user_id
        )
        if not conversation:
            return None
        
        user_message = self.message_repo.create_message(
            conversation_id, "user", message_content
        )
        
        # Use RAG as the default and only pipeline
        digital_human_config = self._get_digital_human_config(
            conversation.digital_human_id
        )
        system_prompt = digital_human_config.get('system_prompt')
        
        try:
            # Use RAG service synchronously
            rag_response = self.rag_service.rag_service.chat_sync(
                user_message=message_content,
                conversation_id=conversation.thread_id,
                user_id=user_id,
                digital_human_id=conversation.digital_human_id,
                system_prompt=system_prompt
            )
            
            ai_message = self.message_repo.create_message(
                conversation_id, "assistant", rag_response.response
            )
            
            # Store additional RAG metadata
            if hasattr(ai_message, 'metadata') and rag_response.metadata:
                ai_message.metadata = rag_response.metadata
            
            print(f"[CONVERSATION_SERVICE] RAG response - Memory: {rag_response.memory_used}, Web: {rag_response.web_results}, Time: {rag_response.processing_time}s")
            
            return MessageResponse.from_orm(ai_message)
            
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise ValueError(f"RAG消息发送失败: {str(e)}")
    
    def send_message_stream(
        self,
        conversation_id: int,
        message_content: str,
        user_id: int
    ) -> Generator[str, None, None]:
        conversation = self.conversation_repo.get_conversation_by_id(
            conversation_id, user_id
        )
        if not conversation:
            yield json.dumps({
                "type": "error",
                "content": "对话不存在或无权限访问"
            })
            return
        
        try:
            user_message = self.message_repo.create_message(
                conversation_id, "user", message_content
            )
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "content": f"保存消息失败: {str(e)}"
            })
            return

        yield json.dumps({
            "type": "message",
            "content": "",
            "metadata": {
                "message_id": user_message.id,
                "role": "user",
                "content": message_content
            }
        })

        # Use RAG streaming as the default and only pipeline
        digital_human_config = self._get_digital_human_config(
            conversation.digital_human_id
        )
        system_prompt = digital_human_config.get('system_prompt')
        
        try:
            # Use async generator from RAG service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                async_generator = self.rag_service.stream_response_async(
                    message=message_content,
                    user_id=user_id,
                    digital_human_id=conversation.digital_human_id,
                    conversation_id=conversation.thread_id,
                    system_prompt=system_prompt
                )
                
                full_response = ""
                
                async def process_stream():
                    nonlocal full_response
                    async for chunk in async_generator:
                        # Parse the SSE format
                        if chunk.startswith("data: "):
                            data_str = chunk[6:].strip()
                            if data_str:
                                try:
                                    data = json.loads(data_str)
                                    if data.get("type") == "chunk":
                                        content = data.get("content", "")
                                        full_response += content
                                        yield json.dumps({
                                            "type": "token",
                                            "content": content
                                        })
                                    elif data.get("type") == "metadata":
                                        yield json.dumps({
                                            "type": "rag_metadata",
                                            "content": "",
                                            "metadata": data
                                        })
                                    elif data.get("type") == "complete":
                                        break
                                    elif data.get("type") == "error":
                                        yield json.dumps({
                                            "type": "error",
                                            "content": data.get("error", "RAG streaming error")
                                        })
                                        return
                                except json.JSONDecodeError:
                                    continue
                
                # Run the async generator
                async_gen = process_stream()
                try:
                    while True:
                        chunk = loop.run_until_complete(async_gen.__anext__())
                        yield chunk
                except StopAsyncIteration:
                    pass
                
                # Save the AI message
                try:
                    ai_message = self.message_repo.create_message(
                        conversation_id, "assistant", full_response
                    )
                except Exception as e:
                    print(f"Warning: Failed to save AI message: {str(e)}")
                    ai_message = None

                yield json.dumps({
                    "type": "done",
                    "content": "",
                    "metadata": {
                        "message_id": ai_message.id if ai_message else None,
                        "tokens_used": ai_message.tokens_used if ai_message else None,
                        "rag_pipeline": True
                    }
                })
                
            finally:
                loop.close()
                
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "content": f"RAG流式响应失败: {str(e)}"
            })
    
    def _get_digital_human_config(self, digital_human_id: int) -> Dict[str, Any]:
        digital_human = self.db.query(DigitalHuman).filter(
            DigitalHuman.id == digital_human_id
        ).first()
        
        if not digital_human:
            return {}
        
        return {
            "name": digital_human.name,
            "type": digital_human.type,
            "skills": digital_human.skills or [],
            "personality": digital_human.personality or {},
            "conversation_style": digital_human.conversation_style,
            "temperature": digital_human.temperature,
            "max_tokens": digital_human.max_tokens,
            "system_prompt": digital_human.system_prompt
        }
    
    def get_conversation_messages(
        self,
        conversation_id: int,
        user_id: int,
        limit: Optional[int] = None
    ) -> List[MessageResponse]:
        conversation = self.conversation_repo.get_conversation_by_id(
            conversation_id, user_id
        )
        if not conversation:
            return []
        
        messages = self.message_repo.get_conversation_messages(
            conversation_id, limit
        )
        
        return [MessageResponse.from_orm(msg) for msg in messages]
    
    def clear_conversation_history(
        self,
        conversation_id: int,
        user_id: int
    ) -> bool:
        conversation = self.conversation_repo.get_conversation_by_id(
            conversation_id, user_id
        )
        if not conversation:
            return False
        
        success = self.message_repo.delete_conversation_messages(conversation_id)
        
        if success:
            # Clear RAG memory
            try:
                # Clear memory asynchronously (we'll use sync wrapper)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        self.rag_service.clear_memory_async(user_id)
                    )
                    print(f"[CONVERSATION_SERVICE] Cleared RAG memory for user {user_id}")
                finally:
                    loop.close()
            except Exception as e:
                print(f"[WARNING] Failed to clear RAG memory: {e}")
        
        return success
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics"""
        try:
            return self.rag_service.get_stats()
        except Exception as e:
            return {"error": str(e), "rag_pipeline": True}
    
    def rag_health_check(self) -> Dict[str, Any]:
        """Perform RAG health check"""
        try:
            return self.rag_service.health_check()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "rag_pipeline": True}