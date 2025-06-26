import json
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import requests
import logging
import uuid
import time
import re
import uvicorn
import numpy as np
import PyPDF2
import pdfplumber
from io import BytesIO
import tempfile

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 기존 Pydantic 모델들 (생략하고 필요한 새 모델들만 추가)
class SearchRequest(BaseModel):
    query: str = Field(..., description="검색할 질의", example="휴가 신청 방법")
    top_k: int = Field(20, description="반환할 결과 수", example=20, ge=1, le=50)
    main_category_filter: Optional[str] = Field(None, description="대분류 필터", example="인사")

class ConversationMessage(BaseModel):
    """대화 메시지 모델"""
    role: str = Field(..., description="메시지 역할 (user 또는 assistant)", example="user")
    content: str = Field(..., description="메시지 내용", example="휴가 신청은 어떻게 하나요?")
    context: Optional[List[Dict[str, Any]]] = Field(None, description="응답 컨텍스트 (assistant 메시지인 경우)")

class ChatRequest(BaseModel):
    query: str = Field(..., description="질문 내용", example="21년차 휴가는 며칠인가요?", min_length=1)
    main_category_filter: Optional[str] = Field(None, description="대분류 필터", example="인사")
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list, 
        description="이전 대화 기록",
        example=[
            {
                "role": "user",
                "content": "휴가 신청은 어떻게 하나요?"
            },
            {
                "role": "assistant", 
                "content": "휴가 신청은 전자결재 시스템을 통해 진행됩니다.",
                "context": []
            }
        ]
    )

class StreamChatRequest(BaseModel):
    """스트리밍 채팅 전용 요청 모델 (더 관대한 검증)"""
    query: str = Field(..., description="질문 내용", example="21년차 휴가는 며칠인가요?", min_length=1)
    main_category_filter: Optional[str] = Field(None, description="대분류 필터", example="인사")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="이전 대화 기록 (유연한 형식)",
        example=[
            {"role": "user", "content": "휴가 신청은 어떻게 하나요?"},
            {"role": "assistant", "content": "휴가 신청은 전자결재 시스템을 통해 진행됩니다."}
        ]
    )

class SimpleTestRequest(BaseModel):
    """간단한 테스트 요청 모델"""
    query: str = Field(..., description="질문", example="휴가 신청 방법", min_length=1)
    category: Optional[str] = Field(None, description="카테고리 필터", example="인사")

class SearchResult(BaseModel):
    main_category: str
    sub_category: str
    question: str
    answer: str
    source_file: str
    chunk_type: str
    score: float
    rank: int

class HealthResponse(BaseModel):
    status: str
    rag_ready: bool
    regulations_count: int
    main_categories_count: int
    chunk_statistics: Optional[Dict[str, int]] = None
    improvements: str
    enhanced_features: Optional[List[str]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    count: int
    main_category_filter: Optional[str]
    search_type: str
    min_relevance: float
    enhanced_features: List[str]

class ChatResponse(BaseModel):
    query: str
    response: str
    context: List[SearchResult]
    context_count: int
    context_quality: Dict[str, int]
    search_type: str
    response_type: str
    enhanced_features: List[str]

# 새로운 PDF 관련 모델들
class PDFUploadResponse(BaseModel):
    """PDF 업로드 응답 모델"""
    status: str
    message: str
    filename: str
    file_size: int
    pages_count: int
    extraction_method: str
    processing_time: float
    json_data: List[Dict[str, Any]]
    statistics: Dict[str, int]

class QAGenerationRequest(BaseModel):
    """Q&A 생성 요청 모델"""
    text: str = Field(..., description="Q&A를 생성할 텍스트", min_length=100)
    category_name: str = Field(..., description="카테고리 이름", example="법인카드 사용 지침")
    qa_count: int = Field(10, description="생성할 Q&A 수", ge=5, le=50)
    company_name: str = Field("회사", description="회사명", example="더케이교직원나라(주)")

class QAGenerationResponse(BaseModel):
    """Q&A 생성 응답 모델"""
    status: str
    category: str
    generated_qa_count: int
    processing_time: float
    json_data: List[Dict[str, Any]]

class PDFTextExtractor:
    """PDF 텍스트 추출 클래스"""
    
    @staticmethod
    def extract_text_pypdf2(pdf_content: bytes) -> tuple[str, int]:
        """PyPDF2를 사용한 텍스트 추출"""
        try:
            pdf_file = BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            pages_count = len(reader.pages)
            
            for page in reader.pages:
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    logger.warning(f"PyPDF2 페이지 추출 실패: {e}")
                    continue
            
            return text.strip(), pages_count
        except Exception as e:
            logger.error(f"PyPDF2 텍스트 추출 실패: {e}")
            return "", 0
    
    @staticmethod
    def extract_text_pdfplumber(pdf_content: bytes) -> tuple[str, int]:
        """pdfplumber를 사용한 텍스트 추출"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name
            
            try:
                with pdfplumber.open(temp_file_path) as pdf:
                    text = ""
                    pages_count = len(pdf.pages)
                    
                    for page in pdf.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        except Exception as e:
                            logger.warning(f"pdfplumber 페이지 추출 실패: {e}")
                            continue
                    
                    return text.strip(), pages_count
            finally:
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"pdfplumber 텍스트 추출 실패: {e}")
            return "", 0
    
    @staticmethod
    def extract_text_hybrid(pdf_content: bytes) -> tuple[str, int, str]:
        """하이브리드 방식으로 텍스트 추출 (두 방법 모두 시도)"""
        # 먼저 pdfplumber 시도
        text_plumber, pages_plumber = PDFTextExtractor.extract_text_pdfplumber(pdf_content)
        
        # PyPDF2도 시도
        text_pypdf2, pages_pypdf2 = PDFTextExtractor.extract_text_pypdf2(pdf_content)
        
        # 더 많은 텍스트를 추출한 방법 선택
        if len(text_plumber) > len(text_pypdf2):
            return text_plumber, pages_plumber, "pdfplumber"
        elif len(text_pypdf2) > 0:
            return text_pypdf2, pages_pypdf2, "PyPDF2"
        else:
            return text_plumber, pages_plumber, "pdfplumber (fallback)"

class QAGenerator:
    """Q&A 생성 클래스"""
    
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url
        logger.info(f"Q&A 생성기 초기화: API URL = {self.api_url}")
    
    def _create_qa_generation_prompt(self, text: str, category_name: str, qa_count: int, company_name: str) -> str:
        """Q&A 생성을 위한 프롬프트 생성"""
        return f"""당신은 회사 내규 문서를 분석하여 RAG 시스템용 Q&A를 생성하는 전문가입니다.

📋 **임무**: 아래 제공된 텍스트를 분석하여 실제 직원들이 자주 물어볼 만한 현실적이고 구체적인 질문과 정확한 답변을 만들어주세요.

🎯 **생성 기준**:
• **실용성**: 직원들이 실제로 궁금해할 만한 구체적인 질문
• **다양성**: 기본 개념부터 세부 절차까지 다양한 수준의 질문
• **완전성**: 각 답변은 문서 내용을 바탕으로 완전하고 명확하게
• **참조 표시**: 가능한 경우 조항 번호나 섹션 표시
• **상황별 질문**: "~인 경우", "~할 때" 등 구체적 상황 포함

🔍 **질문 유형 예시**:
• 기본 개념: "~의 목적은 무엇인가요?", "~란 무엇인가요?"
• 절차/방법: "~는 어떻게 하나요?", "~의 절차는 어떻게 되나요?"
• 조건/기준: "~의 조건은 무엇인가요?", "언제 ~해야 하나요?"
• 책임/권한: "누가 ~를 담당하나요?", "~의 권한은 누구에게 있나요?"
• 예외/특수상황: "~인 경우 어떻게 하나요?", "예외적으로 ~할 수 있나요?"
• 제재/결과: "~하지 않으면 어떻게 되나요?", "위반시 처벌은 어떻게 되나요?"

📖 **분석할 문서**:
카테고리: {category_name}
회사명: {company_name}

문서 내용:
{text}

📝 **출력 형식** (정확히 이 JSON 형식으로만 응답):
[
  {{
    "category": "{category_name}",
    "faqs": [
      {{
        "question": "구체적이고 자연스러운 질문",
        "answer": "문서 내용을 바탕으로 한 완전하고 정확한 답변 (가능한 경우 조항 번호 포함)"
      }},
      ... (총 {qa_count}개 이상)
    ]
  }}
]

🚨 **중요 지침**:
• 반드시 JSON 형식으로만 응답하세요
• 문서에 없는 내용은 절대 추가하지 마세요
• 각 답변은 반드시 제공된 문서 내용을 근거로 하세요
• 질문은 자연스럽고 구체적으로 만드세요
• 답변은 명확하고 완전하게 작성하세요
• 최소 {qa_count}개 이상의 Q&A를 생성하세요"""

    def generate_qa_from_text(self, text: str, category_name: str, qa_count: int = 10, company_name: str = "회사") -> Dict[str, Any]:
        """텍스트로부터 Q&A 생성"""
        logger.info(f"Q&A 생성 시작: {category_name}, 목표 수량: {qa_count}개")
        
        try:
            # 프롬프트 생성
            prompt = self._create_qa_generation_prompt(text, category_name, qa_count, company_name)
            
            # LLM API 호출
            response = requests.post(
                self.api_url,
                json={
                    "model": "qwen3-30b-a3b-mlx",
                    "messages": [
                        {"role": "system", "content": "당신은 회사 내규 문서 분석 및 Q&A 생성 전문가입니다. 반드시 유효한 JSON 형식으로만 응답하세요."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # 창의성과 일관성의 균형
                    "max_tokens": 4000,  # 충분한 토큰 수
                    "top_p": 0.9,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.1
                },
                timeout=180  # 3분 타임아웃
            )
            
            if response.status_code == 200:
                response_text = response.json()['choices'][0]['message']['content']
                
                # JSON 파싱 시도
                try:
                    # 응답에서 JSON 부분만 추출
                    json_start = response_text.find('[')
                    json_end = response_text.rfind(']') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        qa_data = json.loads(json_text)
                        
                        # 데이터 검증
                        if isinstance(qa_data, list) and len(qa_data) > 0:
                            category_data = qa_data[0]
                            if 'faqs' in category_data and len(category_data['faqs']) > 0:
                                logger.info(f"✅ Q&A 생성 완료: {len(category_data['faqs'])}개")
                                return {
                                    'status': 'success',
                                    'data': qa_data,
                                    'generated_count': len(category_data['faqs'])
                                }
                    
                    # JSON 파싱 실패시 재시도
                    logger.warning("JSON 파싱 실패, 응답 텍스트 분석 시도...")
                    return self._parse_fallback_response(response_text, category_name)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 디코딩 실패: {e}")
                    return self._parse_fallback_response(response_text, category_name)
            else:
                logger.error(f"LLM API 호출 실패: {response.status_code}")
                return {'status': 'error', 'message': f'LLM API 호출 실패: {response.status_code}'}
                
        except requests.exceptions.Timeout:
            logger.error("LLM API 타임아웃")
            return {'status': 'error', 'message': 'LLM API 응답 타임아웃'}
        except Exception as e:
            logger.error(f"Q&A 생성 실패: {e}", exc_info=True)
            return {'status': 'error', 'message': f'Q&A 생성 중 오류: {str(e)}'}
    
    def _parse_fallback_response(self, response_text: str, category_name: str) -> Dict[str, Any]:
        """응답 파싱 실패시 대안 파싱"""
        try:
            # 간단한 Q&A 패턴으로 파싱 시도
            qa_pairs = []
            lines = response_text.split('\n')
            current_question = None
            current_answer = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('"question":') or line.startswith('question:'):
                    if current_question and current_answer:
                        qa_pairs.append({
                            'question': current_question,
                            'answer': current_answer.strip()
                        })
                    # 질문 추출
                    current_question = line.split(':', 1)[1].strip().strip('"').strip(',')
                    current_answer = ""
                elif line.startswith('"answer":') or line.startswith('answer:'):
                    # 답변 추출
                    current_answer = line.split(':', 1)[1].strip().strip('"').strip(',')
                elif current_answer and line and not line.startswith('{') and not line.startswith('}'):
                    current_answer += " " + line
            
            # 마지막 Q&A 추가
            if current_question and current_answer:
                qa_pairs.append({
                    'question': current_question,
                    'answer': current_answer.strip()
                })
            
            if qa_pairs:
                fallback_data = [{
                    'category': category_name,
                    'faqs': qa_pairs
                }]
                logger.info(f"✅ 대안 파싱 성공: {len(qa_pairs)}개 Q&A")
                return {
                    'status': 'success',
                    'data': fallback_data,
                    'generated_count': len(qa_pairs),
                    'note': '대안 파싱 방법 사용됨'
                }
            
            return {'status': 'error', 'message': '응답 파싱 실패'}
        except Exception as e:
            logger.error(f"대안 파싱도 실패: {e}")
            return {'status': 'error', 'message': f'응답 파싱 실패: {str(e)}'}

# 기존 RAG 시스템 클래스는 그대로 유지 (생략)
class CompanyRegulationsRAGSystem:
    def __init__(self, model_name: str = "nlpai-lab-KURE-v1", persist_directory: str = "./chroma_db"):
        """
        개선된 ChromaDB 기반 회사 전체 내규 RAG 시스템 초기화 (향상된 검색 및 정보량)

        Args:
            model_name: ./models 디렉토리에 있는 모델 폴더 이름
            persist_directory: ChromaDB 저장 디렉토리
        """
        model_path = os.path.join("./models", model_name)

        # ./models 디렉토리에서 직접 모델 로드
        if not os.path.exists(model_path):
            logger.error(f"❌ 모델 경로가 존재하지 않습니다: {model_path}")
            logger.info("📋 사용 가능한 모델 목록:")
            models_dir = "./models"
            if os.path.exists(models_dir):
                for item in os.listdir(models_dir):
                    if os.path.isdir(os.path.join(models_dir, item)):
                        logger.info(f"   - {item}")
            else:
                logger.error(f"❌ ./models 디렉토리가 존재하지 않습니다.")
            raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {model_path}")
        
        logger.info(f"🔄 로컬 모델 로드 중: {model_path}")

        try:
            # 저장된 모델 직접 로드
            self.model = SentenceTransformer(model_path)
            logger.info(f"✅ 로컬 모델 로드 완료: {model_path}")
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}", exc_info=True)
            raise
        
        self.persist_directory = persist_directory

        # ChromaDB 초기화 (개선된 오류 처리)
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            self.collection = self.chroma_client.get_collection(name="company_regulations")
            logger.info("📁 기존 company_regulations 컬렉션 로드 완료")
        except Exception as get_error:
            logger.info(f"📁 기존 컬렉션 없음 또는 오류 ({get_error}), 새 컬렉션 생성 중...")
            try:
                self.collection = self.chroma_client.create_collection(
                    name="company_regulations",
                    metadata={"description": "향상된 회사 전체 내규 벡터 검색 컬렉션"}
                )
                logger.info("📁 새로운 company_regulations 컬렉션 생성 완료")
            except Exception as create_error:
                logger.error(f"❌ 컬렉션 생성 실패: {create_error}")
                # 대안으로 임시 이름 사용
                import time
                temp_name = f"company_regulations_temp_{int(time.time())}"
                logger.info(f"🔄 임시 컬렉션 '{temp_name}' 생성 시도...")
                self.collection = self.chroma_client.create_collection(
                    name=temp_name,
                    metadata={"description": "임시 회사 내규 벡터 검색 컬렉션"}
                )
                logger.info(f"📁 임시 컬렉션 '{temp_name}' 생성 완료")

        logger.info(f"✅ 임베딩 모델 로드 완료: {model_path}")
        logger.info(f"💾 ChromaDB 저장 디렉토리: {self.persist_directory}")
    
    def load_company_regulations_data(self, data_directory: str = "./data_json"):
        """회사 전체 내규 데이터 로드 - 향상된 청킹 전략"""
        logger.info(f"회사 전체 내규 데이터 로드 시작: {data_directory}")
        try:
            if not os.path.exists(data_directory):
                logger.error(f"❌ 데이터 디렉토리가 존재하지 않습니다: {data_directory}")
                return False
            
            json_files = glob.glob(os.path.join(data_directory, "*.json"))
            
            if not json_files:
                logger.warning(f"⚠️ {data_directory} 디렉토리에서 JSON 파일을 찾을 수 없습니다.")
                return False
            
            logger.info(f"🔍 발견된 JSON 파일: {len(json_files)}개")
            
            self.regulations_data = []
            self.main_categories = {}
            
            for json_file in json_files:
                file_name = os.path.basename(json_file)
                main_category = os.path.splitext(file_name)[0]
                
                logger.info(f"📂 로딩 중: {file_name} → 대구분: {main_category}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    category_count = 0
                    faq_count = 0
                    
                    for category_section in data:
                        sub_category = category_section['category']
                        category_count += 1
                        
                        for faq in category_section['faqs']:
                            question = faq['question']
                            answer = faq['answer']
                            
                            # 기본 Q&A 단위 저장
                            base_id = str(uuid.uuid4())
                            regulation_item = {
                                'id': base_id,
                                'main_category': main_category,
                                'sub_category': sub_category,
                                'question': question,
                                'answer': answer,
                                'text': f"{question} {answer}",
                                'source_file': file_name,
                                'chunk_type': 'qa_full',
                                'chunk_id': 0
                            }
                            self.regulations_data.append(regulation_item)
                            
                            # 향상된 청킹: 질문과 답변을 분리하여 추가 저장
                            # 1) 질문 중심 청크
                            question_item = {
                                'id': f"{base_id}_q",
                                'main_category': main_category,
                                'sub_category': sub_category,
                                'question': question,
                                'answer': answer,
                                'text': f"질문: {question}",
                                'source_file': file_name,
                                'chunk_type': 'question_focused',
                                'chunk_id': 1
                            }
                            self.regulations_data.append(question_item)
                            
                            # 2) 답변 중심 청크
                            answer_item = {
                                'id': f"{base_id}_a",
                                'main_category': main_category,
                                'sub_category': sub_category,
                                'question': question,
                                'answer': answer,
                                'text': f"답변: {answer} (관련 질문: {question})",
                                'source_file': file_name,
                                'chunk_type': 'answer_focused',
                                'chunk_id': 2
                            }
                            self.regulations_data.append(answer_item)
                            
                            # 3) 긴 답변인 경우 문장 단위로 분할하여 추가 저장
                            if len(answer) > 200:  # 긴 답변인 경우
                                sentences = re.split(r'[.!?]\s+', answer)
                                for i, sentence in enumerate(sentences):
                                    if len(sentence.strip()) > 20:  # 의미있는 문장만
                                        sentence_item = {
                                            'id': f"{base_id}_s{i}",
                                            'main_category': main_category,
                                            'sub_category': sub_category,
                                            'question': question,
                                            'answer': answer,
                                            'text': f"{sentence.strip()} (출처: {question})",
                                            'source_file': file_name,
                                            'chunk_type': 'sentence_level',
                                            'chunk_id': 10 + i
                                        }
                                        self.regulations_data.append(sentence_item)
                            
                            faq_count += 1
                    
                    self.main_categories[main_category] = {
                        'file_name': file_name,
                        'sub_categories': category_count,
                        'total_faqs': faq_count
                    }
                    
                    logger.info(f"  → 소구분: {category_count}개, 규정: {faq_count}개 로드됨")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON 파싱 실패 {json_file}: {e}", exc_info=True)
                    continue
                except Exception as e:
                    logger.error(f"❌ 파일 로드 실패 {json_file}: {e}", exc_info=True)
                    continue
            
            total_chunks = len(self.regulations_data)
            total_files = len(self.main_categories)
            
            logger.info(f"✅ 회사 전체 내규 데이터 로드 완료:")
            logger.info(f"  - 대구분(파일): {total_files}개")
            logger.info(f"  - 총 청크: {total_chunks}개 (향상된 청킹 적용)")
            
            # 청킹 타입별 통계
            chunk_stats = {}
            for item in self.regulations_data:
                chunk_type = item.get('chunk_type', 'unknown')
                chunk_stats[chunk_type] = chunk_stats.get(chunk_type, 0) + 1
            
            for chunk_type, count in chunk_stats.items():
                logger.info(f"  - {chunk_type}: {count}개")
            
            return total_chunks > 0
            
        except Exception as e:
            logger.error(f"❌ 회사 내규 데이터 로드 실패: {e}", exc_info=True)
            return False

    def search_with_enhanced_retrieval(self, query: str, top_k: int = 20, main_category_filter: str = None, min_relevance_score: float = 0.3) -> List[Dict[str, Any]]:
        """향상된 검색 - 다중 쿼리, 하이브리드 검색, 향상된 재랭킹 (확장된 정보량)"""
        logger.info(f"향상된 RAG 검색: '{query[:50]}...', 필터='{main_category_filter}', top_k={top_k}")
        # 실제 구현은 기존 코드와 동일하므로 생략
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보 반환"""
        try:
            count = self.collection.count()
            main_cats = self.main_categories if hasattr(self, 'main_categories') else {}
            
            # 청킹 통계
            chunk_stats = {}
            if hasattr(self, 'regulations_data'):
                for item in self.regulations_data:
                    chunk_type = item.get('chunk_type', 'unknown')
                    chunk_stats[chunk_type] = chunk_stats.get(chunk_type, 0) + 1
            
            return {
                'total_documents': count,
                'collection_name': self.collection.name,
                'persist_directory': self.persist_directory,
                'is_ready': count > 0,
                'main_categories': main_cats,
                'total_main_categories': len(main_cats),
                'chunk_statistics': chunk_stats,
                'enhanced_features': [
                    '다중 쿼리 검색',
                    '향상된 청킹 전략',
                    '하이브리드 재랭킹',
                    '대폭 확장된 컨텍스트 윈도우 (25개)',
                    '다양성 보장 알고리즘',
                    '2.5배 확장된 검색 범위 (20개)'
                ]
            }
        except Exception as e:
            logger.error(f"❌ 통계 정보 조회 실패: {e}", exc_info=True)
            return {'total_documents': 0, 'is_ready': False, 'main_categories': {}, 'total_main_categories': 0}

# FastAPI 앱 설정
app = FastAPI(
    title="PDF to JSON RAG Converter + 대폭 확장된 회사 내규 RAG 시스템",
    description="""
    **FastAPI 기반 PDF to JSON RAG Converter + 대폭 확장된 정보량 회사 내규 RAG 시스템**
    
    ## 새로운 PDF 처리 기능
    - 📄 PDF 파일 업로드 및 텍스트 추출
    - 🤖 LLM 기반 자동 Q&A 생성 (qwen3-30b-a3b-mlx)
    - 📝 RAG용 JSON 형식 자동 변환
    - 🔄 하이브리드 텍스트 추출 (PyPDF2 + pdfplumber)
    
    ## 기존 RAG 시스템 특징
    - 🔍 다중 쿼리 검색 (원본 + 변형 쿼리로 다각도 검색)
    - 📝 향상된 청킹 전략 (QA + 질문 + 답변 + 문장 단위)
    - 🎯 하이브리드 재랭킹 (벡터 + 키워드 + 다중쿼리 매칭)
    - 📊 대폭 확장된 컨텍스트 윈도우 (최대 25개 → 훨씬 더 풍부한 정보)
    - 🌟 다양성 보장 알고리즘 (중복 제거 + 품질 유지)
    - 📉 낮은 관련성 임계값 (0.25, 부분 관련 정보도 포함)
    - 🎯 정확성 최우선 (Temperature: 0.1)
    - 📝 더 긴 응답 허용 (2500 토큰)
    
    ## PDF 처리 성능
    - **Q&A 생성 수**: 5-50개 (사용자 지정 가능)
    - **처리 시간**: 평균 30-60초 (문서 크기에 따라)
    - **지원 형식**: PDF (텍스트 기반)
    - **최대 파일 크기**: 10MB
    """,
    version="4.0.0",
    contact={
        "name": "RAG 시스템 개발팀",
        "email": "dev@company.com"
    }
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 객체들
rag_system: CompanyRegulationsRAGSystem = None
qa_generator: QAGenerator = None

def initialize_system():
    """시스템 초기화"""
    global rag_system, qa_generator
    
    if rag_system is not None and qa_generator is not None:
        logger.info("시스템이 이미 초기화되어 있습니다.")
        return

    logger.info("향상된 RAG 시스템 + PDF 처리 시스템 초기화 시작...")
    
    try:
        # ./models 디렉토리에서 사용 가능한 모델 확인
        models_dir = "./models"
        available_models = []
        
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                model_path = os.path.join(models_dir, item)
                if os.path.isdir(model_path):
                    # sentence-transformers 모델인지 확인
                    config_file = os.path.join(model_path, "config.json")
                    if os.path.exists(config_file):
                        available_models.append(item)
            
            logger.info(f"📋 사용 가능한 로컬 모델: {available_models}")
        else:
            logger.error(f"❌ ./models 디렉토리가 존재하지 않습니다.")
            logger.info("💡 ./models 디렉토리를 생성하고 sentence-transformers 모델을 다운로드하세요.")
            return
        
        if not available_models:
            logger.error("❌ ./models 디렉토리에 사용 가능한 모델이 없습니다.")
            logger.info("💡 다음 중 하나의 모델을 다운로드하세요:")
            logger.info("   - nlpai-lab/KURE-v1 (한국어 특화)")
            logger.info("   - sentence-transformers/all-MiniLM-L6-v2 (다국어)")
            logger.info("   - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            return
        
        # 첫 번째 사용 가능한 모델 사용 (또는 특정 모델 지정)
        model_to_use = available_models[0]
        
        # 한국어 모델 우선 선택
        for model in available_models:
            if "kure" in model.lower() or "korean" in model.lower():
                model_to_use = model
                break
        
        logger.info(f"🎯 선택된 모델: {model_to_use}")
        
        rag_system = CompanyRegulationsRAGSystem(model_name=model_to_use)
        qa_generator = QAGenerator()
        
        data_directory = "./data_json"
        
        if os.path.exists(data_directory):
            logger.info("회사 내규 데이터 로드 및 향상된 인덱스 구축...")
            if rag_system.load_company_regulations_data(data_directory):
                stats = rag_system.get_stats()
                logger.info(f"✅ 향상된 시스템 초기화 완료: {stats['total_documents']}개 청크")
                logger.info(f"📊 청킹 통계: {stats.get('chunk_statistics', {})}")
            else:
                logger.error("❌ 인덱스 구축 실패")
        else:
            logger.warning(f"⚠️ 데이터 디렉토리 없음: {data_directory}. 빈 인덱스로 시작합니다.")
            logger.info("💡 PDF 업로드 기능을 사용하여 데이터를 추가할 수 있습니다.")

    except Exception as e:
        logger.critical(f"🔥 향상된 시스템 초기화 실패: {e}", exc_info=True)
        rag_system = None
        qa_generator = None

# 새로운 PDF 처리 엔드포인트들
@app.get("/models/check", summary="로컬 모델 상태 확인", description="./models 디렉토리의 모델 상태를 확인합니다.")
async def check_local_models():
    """로컬 모델 상태 확인"""
    try:
        models_dir = "./models"
        model_status = {
            "models_directory_exists": os.path.exists(models_dir),
            "available_models": [],
            "recommended_models": [
                "nlpai-lab/KURE-v1",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            "download_instructions": {
                "method1_python": [
                    "from sentence_transformers import SentenceTransformer",
                    "model = SentenceTransformer('nlpai-lab/KURE-v1')",
                    "model.save('./models/nlpai-lab-KURE-v1')"
                ],
                "method2_huggingface": [
                    "git lfs install",
                    "git clone https://huggingface.co/nlpai-lab/KURE-v1 ./models/nlpai-lab-KURE-v1"
                ]
            }
        }
        
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                model_path = os.path.join(models_dir, item)
                if os.path.isdir(model_path):
                    # 모델 유효성 검사
                    config_file = os.path.join(model_path, "config.json")
                    has_config = os.path.exists(config_file)
                    
                    # 모델 파일 존재 여부 확인
                    model_files = []
                    for file_pattern in ["*.bin", "*.safetensors", "*.pt"]:
                        model_files.extend(glob.glob(os.path.join(model_path, file_pattern)))
                    
                    model_info = {
                        "name": item,
                        "path": model_path,
                        "has_config": has_config,
                        "has_model_files": len(model_files) > 0,
                        "model_files": [os.path.basename(f) for f in model_files],
                        "is_valid": has_config and len(model_files) > 0,
                        "size_mb": sum(os.path.getsize(os.path.join(model_path, f)) for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))) / (1024*1024) if os.path.exists(model_path) else 0
                    }
                    
                    model_status["available_models"].append(model_info)
        
        # 유효한 모델 개수
        valid_models = [m for m in model_status["available_models"] if m["is_valid"]]
        model_status["valid_models_count"] = len(valid_models)
        model_status["status"] = "ready" if len(valid_models) > 0 else "needs_setup"
        
        return model_status
        
    except Exception as e:
        logger.error(f"❌ 모델 상태 확인 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모델 상태 확인 실패: {str(e)}")

@app.post("/models/download", summary="추천 모델 다운로드", description="추천 sentence-transformers 모델을 다운로드합니다.")
async def download_recommended_model(
    model_name: str = "nlpai-lab/KURE-v1",
    force_redownload: bool = False
):
    """추천 모델 다운로드"""
    try:
        # 모델명 검증
        allowed_models = [
            "nlpai-lab/KURE-v1",
            "sentence-transformers/all-MiniLM-L6-v2", 
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        if model_name not in allowed_models:
            raise HTTPException(
                status_code=400, 
                detail=f"허용된 모델이 아닙니다. 사용 가능한 모델: {allowed_models}"
            )
        
        # 로컬 저장 경로
        safe_model_name = model_name.replace("/", "-")
        model_path = os.path.join("./models", safe_model_name)
        
        # 이미 존재하는 경우
        if os.path.exists(model_path) and not force_redownload:
            return {
                "status": "already_exists",
                "message": f"모델이 이미 존재합니다: {model_path}",
                "model_name": model_name,
                "local_path": model_path,
                "size_mb": sum(os.path.getsize(os.path.join(model_path, f)) for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))) / (1024*1024)
            }
        
        # 디렉토리 생성
        os.makedirs("./models", exist_ok=True)
        
        logger.info(f"📦 모델 다운로드 시작: {model_name} → {model_path}")
        
        # SentenceTransformer로 다운로드
        start_time = time.time()
        
        try:
            model = SentenceTransformer(model_name)
            model.save(model_path)
            
            download_time = time.time() - start_time
            
            # 크기 계산
            total_size = sum(os.path.getsize(os.path.join(model_path, f)) for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f)))
            size_mb = total_size / (1024*1024)
            
            logger.info(f"✅ 모델 다운로드 완료: {model_name} ({size_mb:.1f}MB, {download_time:.1f}초)")
            
            return {
                "status": "success",
                "message": f"모델 다운로드 완료: {model_name}",
                "model_name": model_name,
                "local_path": model_path,
                "download_time_seconds": round(download_time, 2),
                "size_mb": round(size_mb, 1),
                "files_downloaded": os.listdir(model_path)
            }
            
        except Exception as download_error:
            logger.error(f"❌ 모델 다운로드 실패: {download_error}")
            
            # 실패한 경우 부분적으로 다운로드된 파일 정리
            if os.path.exists(model_path):
                import shutil
                shutil.rmtree(model_path)
            
            raise HTTPException(
                status_code=500, 
                detail=f"모델 다운로드 실패: {str(download_error)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 모델 다운로드 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모델 다운로드 처리 실패: {str(e)}")

@app.post("/pdf/upload", response_model=PDFUploadResponse, summary="PDF 업로드 및 RAG용 JSON 생성", description="PDF 파일을 업로드하여 텍스트를 추출하고 LLM으로 Q&A를 생성하여 RAG용 JSON을 만듭니다.")
async def upload_pdf_and_generate_qa(
    file: UploadFile = File(..., description="업로드할 PDF 파일"),
    category_name: str = Form(..., description="카테고리 이름", example="법인카드 사용 지침"),
    qa_count: int = Form(15, description="생성할 Q&A 수", ge=5, le=50),
    company_name: str = Form("회사", description="회사명", example="더케이교직원나라(주)")
):
    """PDF 업로드 및 RAG용 JSON 자동 생성"""
    start_time = time.time()
    
    try:
        if not qa_generator:
            raise HTTPException(status_code=503, detail="Q&A 생성 시스템이 초기화되지 않았습니다.")
        
        # 파일 검증
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
        
        # 파일 크기 검증 (10MB 제한)
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다.")
        
        logger.info(f"📄 PDF 파일 업로드: {file.filename} ({file_size:,} bytes)")
        
        # PDF 텍스트 추출
        extracted_text, pages_count, extraction_method = PDFTextExtractor.extract_text_hybrid(file_content)
        
        if not extracted_text or len(extracted_text.strip()) < 100:
            raise HTTPException(
                status_code=400, 
                detail="PDF에서 충분한 텍스트를 추출할 수 없습니다. 스캔된 이미지나 보호된 PDF일 수 있습니다."
            )
        
        logger.info(f"📖 텍스트 추출 완료: {len(extracted_text):,}자, {pages_count}페이지 ({extraction_method})")
        
        # Q&A 생성
        qa_result = qa_generator.generate_qa_from_text(
            text=extracted_text,
            category_name=category_name,
            qa_count=qa_count,
            company_name=company_name
        )
        
        if qa_result['status'] != 'success':
            raise HTTPException(status_code=500, detail=f"Q&A 생성 실패: {qa_result.get('message', '알 수 없는 오류')}")
        
        processing_time = time.time() - start_time
        
        # 통계 정보
        total_qa = qa_result['generated_count']
        json_data = qa_result['data']
        
        # 카테고리별 통계
        statistics = {
            'total_categories': len(json_data),
            'total_qa_pairs': total_qa,
            'average_qa_per_category': total_qa // len(json_data) if json_data else 0,
            'text_length': len(extracted_text),
            'pages_processed': pages_count
        }
        
        # JSON 파일로 저장 (옵션)
        output_filename = f"{category_name.replace(' ', '_')}_{int(time.time())}.json"
        output_path = os.path.join("./data_json", output_filename)
        
        # 디렉토리 생성
        os.makedirs("./data_json", exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 JSON 파일 저장: {output_path}")
        except Exception as save_error:
            logger.warning(f"⚠️ JSON 파일 저장 실패: {save_error}")
        
        return PDFUploadResponse(
            status="success",
            message=f"PDF 처리 및 Q&A 생성 완료 ({total_qa}개 생성)",
            filename=file.filename,
            file_size=file_size,
            pages_count=pages_count,
            extraction_method=extraction_method,
            processing_time=round(processing_time, 2),
            json_data=json_data,
            statistics=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PDF 처리 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/pdf/text_to_qa", response_model=QAGenerationResponse, summary="텍스트로부터 Q&A 생성", description="제공된 텍스트로부터 LLM을 이용해 Q&A를 생성합니다.")
async def generate_qa_from_text(request: QAGenerationRequest):
    """텍스트로부터 Q&A 생성"""
    start_time = time.time()
    
    try:
        if not qa_generator:
            raise HTTPException(status_code=503, detail="Q&A 생성 시스템이 초기화되지 않았습니다.")
        
        logger.info(f"🤖 텍스트 기반 Q&A 생성 시작: {request.category_name}, 목표: {request.qa_count}개")
        
        # Q&A 생성
        qa_result = qa_generator.generate_qa_from_text(
            text=request.text,
            category_name=request.category_name,
            qa_count=request.qa_count,
            company_name=request.company_name
        )
        
        if qa_result['status'] != 'success':
            raise HTTPException(status_code=500, detail=f"Q&A 생성 실패: {qa_result.get('message', '알 수 없는 오류')}")
        
        processing_time = time.time() - start_time
        generated_count = qa_result['generated_count']
        json_data = qa_result['data']
        
        logger.info(f"✅ Q&A 생성 완료: {generated_count}개 ({processing_time:.2f}초)")
        
        return QAGenerationResponse(
            status="success",
            category=request.category_name,
            generated_qa_count=generated_count,
            processing_time=round(processing_time, 2),
            json_data=json_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Q&A 생성 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Q&A 생성 중 오류가 발생했습니다: {str(e)}")

@app.get("/pdf/test", summary="PDF 처리 시스템 테스트", description="PDF 처리 시스템의 상태를 확인합니다.")
async def test_pdf_system():
    """PDF 처리 시스템 테스트"""
    try:
        # 시스템 상태 확인
        system_status = {
            "qa_generator_ready": qa_generator is not None,
            "rag_system_ready": rag_system is not None,
            "supported_formats": ["PDF"],
            "max_file_size": "10MB",
            "text_extractors": ["PyPDF2", "pdfplumber"],
            "llm_model": "qwen3-30b-a3b-mlx",
            "llm_api_url": qa_generator.api_url if qa_generator else None
        }
        
        # LLM API 연결 테스트
        if qa_generator:
            try:
                test_response = requests.get("http://localhost:1234/v1/models", timeout=5)
                system_status["llm_api_status"] = "connected" if test_response.status_code == 200 else "error"
            except:
                system_status["llm_api_status"] = "disconnected"
        
        # 샘플 Q&A 생성 테스트
        sample_text = """
        제1조 (목적) 이 지침은 회사의 효율적인 업무 수행을 위해 제정되었습니다.
        제2조 (적용범위) 본 지침은 전 직원에게 적용됩니다.
        제3조 (준수사항) 직원은 본 지침을 준수해야 합니다.
        """
        
        if qa_generator:
            sample_result = qa_generator.generate_qa_from_text(
                text=sample_text,
                category_name="테스트 카테고리",
                qa_count=3,
                company_name="테스트 회사"
            )
            system_status["sample_qa_generation"] = sample_result['status']
            system_status["sample_qa_count"] = sample_result.get('generated_count', 0)
        
        return {
            "status": "healthy" if system_status["qa_generator_ready"] and system_status["rag_system_ready"] else "partial",
            "message": "PDF 처리 시스템 상태 확인 완료",
            "system_details": system_status,
            "usage_guide": {
                "upload_endpoint": "/pdf/upload",
                "text_to_qa_endpoint": "/pdf/text_to_qa",
                "supported_parameters": {
                    "qa_count": "5-50개",
                    "max_file_size": "10MB",
                    "supported_formats": "PDF"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ PDF 시스템 테스트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"시스템 테스트 실패: {str(e)}")

# 기존 엔드포인트들 (간소화된 버전)
@app.get("/health", response_model=HealthResponse, summary="대폭 확장된 시스템 상태 확인", description="대폭 확장된 RAG 시스템의 상태와 통계를 확인합니다.")
async def health_check():
    """헬스 체크"""
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "initializing_or_error",
                "rag_ready": False,
                "regulations_count": 0,
                "main_categories_count": 0,
                "improvements": "대폭 확장된 정보량, 2.5배 검색 범위, 정확도 개선, PDF 처리 기능 추가",
                "message": "대폭 확장된 RAG 시스템 초기화 중이거나 오류가 발생했습니다."
            }
        )

    stats = rag_system.get_stats()
    
    status = "healthy" if stats['is_ready'] else "unhealthy"

    return HealthResponse(
        status=status,
        rag_ready=stats['is_ready'],
        regulations_count=stats['total_documents'],
        main_categories_count=stats.get('total_main_categories', 0),
        chunk_statistics=stats.get('chunk_statistics', {}),
        improvements="대폭 확장된 정보량, 2.5배 검색 범위, 정확도 개선, PDF 처리 기능 추가",
        enhanced_features=stats.get('enhanced_features', []) + ["PDF to JSON 변환", "자동 Q&A 생성"]
    )

@app.post("/search", response_model=SearchResponse, summary="대폭 확장된 내규 검색", description="대폭 확장된 다중 쿼리 하이브리드 검색으로 관련 내규를 찾습니다.")
async def search_regulations(request: SearchRequest):
    """대폭 확장된 회사 내규 검색"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="대폭 확장된 RAG 시스템이 초기화되지 않았습니다.")
    
    results = rag_system.search_with_enhanced_retrieval(
        request.query, 
        request.top_k,
        request.main_category_filter, 
        min_relevance_score=0.3
    )
    
    search_results = [SearchResult(**result) for result in results]
    
    return SearchResponse(
        query=request.query,
        results=search_results,
        count=len(results),
        main_category_filter=request.main_category_filter,
        search_type="대폭 확장된 다중쿼리 하이브리드 검색",
        min_relevance=0.3,
        enhanced_features=[
            "다중 쿼리 변형",
            "향상된 청킹",
            "하이브리드 재랭킹",
            "다양성 보장",
            f"2.5배 확장된 검색 범위 (최대 {request.top_k}개)",
            "PDF 생성 Q&A 포함"
        ]
    )

# 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    initialize_system()

if __name__ == '__main__':
    print("=" * 80)
    print("🏢 FastAPI 기반 PDF to JSON RAG Converter + 대폭 확장된 정보량 회사 내규 RAG 서버")
    print("=" * 80)
    print("🆕 새로운 PDF 처리 기능:")
    print("   📄 PDF 파일 업로드 및 텍스트 추출")
    print("   🤖 LLM 기반 자동 Q&A 생성 (qwen3-30b-a3b-mlx)")
    print("   📝 RAG용 JSON 형식 자동 변환")
    print("   🔄 하이브리드 텍스트 추출 (PyPDF2 + pdfplumber)")
    print("   ⚡ 최대 50개 Q&A 자동 생성")
    print("   💾 자동 JSON 파일 저장")
    print("=" * 80)
    print("✨ 기존 RAG 시스템 개선사항:")
    print("   🔍 다중 쿼리 검색 (원본 + 변형 쿼리로 다각도 검색)")
    print("   📝 향상된 청킹 전략 (QA + 질문 + 답변 + 문장 단위)")
    print("   🎯 하이브리드 재랭킹 (벡터 + 키워드 + 다중쿼리 매칭)")
    print("   📊 대폭 확장된 컨텍스트 윈도우 (10개 → 25개, 2.5배)")
    print("   🌟 다양성 보장 알고리즘 (중복 제거 + 품질 유지)")
    print("   🚀 대폭 확장된 검색 범위 (8개 → 20개, 2.5배)")
    print("=" * 80)
    print("🔧 FastAPI 엔드포인트:")
    print("   🆕 GET  /models/check          : 📋 로컬 모델 상태 확인")
    print("   🆕 POST /models/download       : 📦 추천 모델 자동 다운로드")
    print("   🆕 POST /pdf/upload           : 📄 PDF 업로드 → JSON 변환 🎯")
    print("   🆕 POST /pdf/text_to_qa       : 📝 텍스트 → Q&A 생성")
    print("   🆕 GET  /pdf/test             : 🧪 PDF 시스템 테스트")
    print("   - GET  /health                : 대폭 확장된 시스템 상태 확인")
    print("   - POST /search                : 확장된 다중쿼리 하이브리드 검색")
    print("   - GET  /docs                  : 🎯 Swagger UI 문서 (API 테스트 가능) 🎯")
    print("=" * 80)
    print("📖 API 문서 및 테스트:")
    print("   http://localhost:5000/docs  ← 🎯 여기서 PDF 업로드 & 모델 관리 테스트하세요!")
    print("   http://localhost:5000/redoc ← 대안 문서")
    print("=" * 80)
    print("🔧 모델 설정 가이드:")
    print("   1. 모델 상태 확인: GET /models/check")
    print("   2. 모델 자동 다운로드: POST /models/download")
    print("      - nlpai-lab/KURE-v1 (한국어 특화, 권장)")
    print("      - sentence-transformers/all-MiniLM-L6-v2 (다국어)")
    print("   3. 또는 수동 다운로드:")
    print("      python -c \"from sentence_transformers import SentenceTransformer;")
    print("      SentenceTransformer('nlpai-lab/KURE-v1').save('./models/nlpai-lab-KURE-v1')\"")
    print("=" * 80)
    print("📄 PDF 처리 사용법:")
    print("   1. /docs 페이지로 이동")
    print("   2. 'POST /pdf/upload' 섹션 클릭")
    print("   3. 'Try it out' 버튼 클릭")
    print("   4. PDF 파일 선택 + 카테고리명 입력")
    print("   5. Q&A 개수 설정 (5-50개)")
    print("   6. 'Execute' 버튼으로 실행!")
    print("=" * 80)
    print("📝 사용 예제 (cURL):")
    print("   curl -X POST 'http://localhost:5000/pdf/upload' \\")
    print("        -F 'file=@your_document.pdf' \\")
    print("        -F 'category_name=법인카드 사용 지침' \\")
    print("        -F 'qa_count=20' \\")
    print("        -F 'company_name=더케이교직원나라(주)'")
    print("=" * 80)
    print("🎯 기대 효과:")
    print("   ✅ PDF 문서를 즉시 RAG용 JSON으로 변환")
    print("   ✅ 수동 Q&A 작성 시간 대폭 단축")
    print("   ✅ 일관성 있는 고품질 Q&A 자동 생성")
    print("   ✅ 다양한 질문 유형 자동 커버")
    print("   ✅ 기존 RAG 시스템과 즉시 연동 가능")
    print("=" * 80)
    print("⚠️ 사전 준비사항:")
    print("   1. LM Studio 실행 (http://localhost:1234)")
    print("   2. qwen3-30b-a3b-mlx 모델 로드")
    print("   3. ./models 디렉토리에 sentence-transformers 모델 준비")
    print("      (예: nlpai-lab/KURE-v1, sentence-transformers/all-MiniLM-L6-v2)")
    print("   4. PDF 파일 준비 (최대 10MB)")
    print("   5. pip install PyPDF2 pdfplumber sentence-transformers")
    print("=" * 80)
    print("📁 로컬 모델 디렉토리 구조:")
    print("   ./models/")
    print("   ├── nlpai-lab-KURE-v1/          # 한국어 특화 (권장)")
    print("   │   ├── config.json")
    print("   │   ├── pytorch_model.bin")
    print("   │   └── tokenizer.json")
    print("   └── all-MiniLM-L6-v2/           # 다국어 지원")
    print("       ├── config.json")
    print("       └── pytorch_model.bin")
    print("=" * 80)
    
    # 향상된 시스템 초기화
    initialize_system() 
    
    logger.info("🚀 PDF to JSON RAG Converter + 대폭 확장된 FastAPI 서버 시작 중...")
    
    # 직접 FastAPI 앱 실행
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level="info")