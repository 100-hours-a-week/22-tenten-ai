from typing import List
import logging
from schemas.bot_chats_schema import BotChatsRequest, BotChatsResponse, BotChatResponseData, UserInfoResponse, BotChatQueueRequest
import os
from models.model_loader import ModelLoader
from langchain.memory import ConversationBufferWindowMemory # ConversationBufferWindowMemory로 변경
import json
from datetime import datetime
import pytz # pytz 임포트
from core.sse_manager import sse_manager
import asyncio # 테스트를 위한 asyncio 임포트
from core.prompt_templates.bot_chats_prompt import BotChatsPrompt # 프롬프트 클라이언트 임포트
from utils.logger import log_inference_to_langfuse # 로거 임포트

class BotChatsService:
    def __init__(self, app):
        """
        BotChatsService 생성자
        - FastAPI app의 state에서 모델 싱글턴 인스턴스를 받아옴
        - stream_id별 대화 메모리(ConversationBufferWindowMemory) 딕셔너리 초기화
        - 채팅 프롬프트 클라이언트 초기화
        """
        self.logger = logging.getLogger(__name__)
        self.model = app.state.model
        self.memory_dict = {}  # key: stream_id, value: ConversationBufferWindowMemory
        self.memory_k = 5  # 최근 5개 메시지만 유지
        self.prompt_client = BotChatsPrompt() # 프롬프트 클라이언트 인스턴스 생성

    def get_memory(self, stream_id: str):
        """
        stream_id별 ConversationBufferWindowMemory 인스턴스를 반환하거나 새로 생성
        Args:
            stream_id (str): 대화 스트림 ID
        Returns:
            ConversationBufferWindowMemory: 해당 stream_id의 메모리 인스턴스
        """
        if stream_id not in self.memory_dict:
            # ConversationBufferWindowMemory를 사용하도록 수정
            self.memory_dict[stream_id] = ConversationBufferWindowMemory(k=self.memory_k, return_messages=True)
        return self.memory_dict[stream_id]

    def add_message_to_memory(self, stream_id: str, role: str, content: str):
        """
        stream_id별 메모리에 메시지를 추가
        Args:
            stream_id (str): 대화 스트림 ID
            role (str): 'user' 또는 'ai' 등 역할
            content (str): 메시지 내용
        """
        memory = self.get_memory(stream_id)
        if role == 'user':
            memory.chat_memory.add_user_message(content)
        elif role == 'ai':
            memory.chat_memory.add_ai_message(content)

    def get_recent_messages(self, stream_id: str):
        """
        stream_id별 최근 대화 기록(최대 k개)을 반환
        Returns:
            List[dict]: [{"role": ..., "content": ...}]
        """
        memory = self.get_memory(stream_id)
        retained_messages = memory.buffer_as_messages

        # [REFACTOR] Langchain의 role(human, ai)을 모델 표준(user, assistant)으로 변환
        role_map = {"human": "user", "ai": "assistant"}
        return [
            {"role": role_map.get(m.type, m.type), "content": m.content}
            for m in retained_messages
        ]

    def delete_memory(self, stream_id: str):
        """
        stream_id별 ConversationBufferMemory 인스턴스를 메모리에서 삭제 (중단/파기)
        Args:
            stream_id (str): 대화 스트림 ID
        """
        if stream_id in self.memory_dict:
            del self.memory_dict[stream_id]


    async def process_chat_and_broadcast(self, request: BotChatQueueRequest):
        """
        채팅을 처리하고, 생성된 응답을 SSEManager를 통해 브로드캐스트
        """
        stream_id = request.stream_id
        KST = pytz.timezone('Asia/Seoul') # KST 시간대 정의
        
        try:
            # [REFACTOR] User 메시지 형식을 다른 기능과 통일
            user_message_content = f"[{request.nickname} from {request.class_name}] {request.message}"
            self.add_message_to_memory(stream_id, "user", user_message_content)

            recent_messages = self.get_recent_messages(stream_id)
            messages_with_persona = self.prompt_client.get_messages_with_persona(recent_messages)

            # [REFACTOR] Langfuse 트레이스를 프롬프트 생성 이후로 이동하고, input에 전체 대화 기록을 추가
            trace = self.prompt_client.langfuse.trace(
                name="bot_chats_streaming_flow",
                metadata={
                    "stream_id": stream_id,
                    "user_id": request.user_id,
                    "nickname": request.nickname
                },
                input=recent_messages, # recent_messages를 input으로 기록
                environment=os.environ.get("LLM_MODE", "colab")
            )

            # Generation 시작 시간 기록
            start_time = datetime.now()
            
            model_response = self.model.get_response(
                messages=messages_with_persona, 
                trace=trace, # model_loader.get_response 시그니처에 맞게 trace 전달
                adapter_type="social_bot",
                endpoint_info="sns_chat"
            )
            ai_content = model_response.get("content", "")

            # Langfuse에 Generation 상세 정보 기록
            log_inference_to_langfuse(
                trace=trace,
                name="bot_chat_generation",
                prompt=self.prompt_client.langfuse.get_prompt("chats_bot"),
                messages=messages_with_persona,
                content=ai_content,
                model_name=self.model.loader.model_path,
                model_parameters={
                    "temperature": self.model.loader.temperature,
                    "top_p": self.model.loader.top_p,
                    "max_tokens": self.model.loader.max_tokens,
                },
                start_time=start_time,
                end_time=datetime.now(),
                inference_time=(datetime.now() - start_time).total_seconds()
            )

            # 단어 단위 스트리밍으로 변경
            for token in ai_content.split(' '):
                # 단어가 비어있지 않은 경우에만 전송
                if not token:
                    continue
                
                stream_data = {
                    "stream_id": stream_id,
                    "message": token + " ", # 각 단어 뒤에 공백을 붙여서 전송
                    "timestamp": datetime.now(KST).strftime('%Y-%m-%dT%H:%M:%S')
                }
                self.logger.info(f"Broadcasting to {stream_id}: {json.dumps(stream_data, ensure_ascii=False)}")
                await sse_manager.broadcast(f"event: stream\ndata: {json.dumps(stream_data, ensure_ascii=False)}\n\n")
                # await asyncio.sleep(0.1) # [TEST] 스트리밍 효과를 확인하기 위한 0.1초 지연

            self.add_message_to_memory(stream_id, "ai", ai_content)
            
            done_data = {
                "stream_id": stream_id,
                "message": None,
                "timestamp": datetime.now(KST).strftime('%Y-%m-%dT%H:%M:%S')
            }
            await sse_manager.broadcast(f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n")

            # Langfuse 트레이스 업데이트 (성공)
            trace.update(output={"full_response": ai_content, "status": "success"})

        except Exception as e:
            self.logger.error(f"Error processing chat for stream {stream_id}: {e}")
            self.delete_memory(stream_id)
            error_data = {
                "stream_id": stream_id,
                "message": str(e),
                "timestamp": datetime.now(KST).strftime('%Y-%m-%dT%H:%M:%S')
            }
            self.logger.error(f"Broadcasting error to {stream_id}: {json.dumps(error_data, ensure_ascii=False)}")
            await sse_manager.broadcast(f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n")
            
            # Langfuse 트레이스 업데이트 (실패)
            if 'trace' in locals():
                trace.update(output={"error": str(e), "status": "error"})
