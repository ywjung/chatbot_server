import json
import os
import glob
import logging
import requests
import uuid
import re
import urllib.parse
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import fitz  # PyMuPDF for PDF processing
from datetime import datetime
import pytz

# 벡터 DB 관련 imports
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI 앱 설정
app = FastAPI(
    title="회사 내규 PDF JSON 추출 + 벡터 DB 통합 시스템",
    description="""
    **회사 내규 PDF에서 구조화된 JSON을 추출하고 벡터 DB에 저장하는 통합 AI 시스템**
    
    ## 주요 기능
    - 📄 **PDF 업로드 및 파싱**: 복잡한 내규 문서 자동 처리
    - 🧠 **AI 기반 구조 분석**: LLM을 활용한 정교한 내용 분석
    - 📊 **표준 JSON 구조**: 기존 내규 형식과 일치하는 구조화된 출력
    - 🎯 **질문-답변 자동 생성**: 모든 내규 조항을 질문-답변 형태로 변환
    - 📝 **카테고리 자동 분류**: 내용 기반 스마트 카테고리 구분
    - 💾 **파일 관리**: JSON 저장, 목록 조회, 다운로드 기능
    - 🔍 **벡터 DB 저장**: RAG 서버와 동일한 형식으로 ChromaDB에 저장
    - 🎯 **벡터 검색**: 고성능 의미 기반 검색 지원
    
    ## AI 분석 특징
    - **포괄적 질문 생성**: 조항별, 절차별, 예외사항별 다각도 질문 생성
    - **정확한 답변 추출**: 원문 기반 정확하고 완전한 답변 제공
    - **구조적 분석**: 계층적 문서 구조 인식 및 반영
    - **맥락 인식**: 연관 조항 간의 관계 파악 및 통합
    
    ## 벡터 DB 기능 (RAG 서버 호환)
    - **ChromaDB 통합**: RAG 서버와 동일한 형식으로 데이터 저장
    - **청킹 전략**: qa_full, question_focused, answer_focused, sentence_level
    - **임베딩 생성**: SentenceTransformer 기반 한국어 임베딩
    - **메타데이터 관리**: RAG 서버와 완전 호환되는 메타데이터 구조
    
    ## API 사용법
    1. `/api/extract_regulation_json`: PDF 업로드 → JSON 추출
    2. `/api/extract_regulation_json_v2`: PDF 업로드 → JSON 추출 (벡터 DB 저장용)
    3. `/api/save_extracted_json_v2`: JSON Body로 저장 (권장)
    4. `/api/store_json_to_vector`: 저장된 JSON을 벡터 DB에 저장 (RAG 서버 호환)
    5. `/api/store_extracted_json_to_vector`: 추출된 JSON을 직접 벡터 DB에 저장 (RAG 서버 호환)
    6. `/api/vector_search`: 벡터 DB에서 의미 기반 검색
    7. `/api/vector_stats`: 벡터 DB 통계 정보 조회
    """,
    version="3.1.0-rag-compatible",
    contact={
        "name": "내규 JSON 추출 + 벡터 DB 통합 시스템 (RAG 서버 호환)",
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

# Pydantic 모델들
class RegulationFAQ(BaseModel):
    question: str = Field(..., description="내규 관련 질문")
    answer: str = Field(..., description="질문에 대한 답변")

class RegulationCategory(BaseModel):
    category: str = Field(..., description="카테고리명 (한글 + 영문)")
    faqs: List[RegulationFAQ] = Field(..., description="질문-답변 목록")

class PDFExtractionRequest(BaseModel):
    regulation_type: Optional[str] = Field("일반내규", description="내규 유형 (예: 직제규정, 인사규정, 급여규정 등)")
    extract_mode: Optional[str] = Field("comprehensive", description="추출 모드 (comprehensive: 포괄적, focused: 핵심만)")

class PDFExtractionResponse(BaseModel):
    success: bool
    regulation_type: str
    categories_count: int
    total_faqs: int
    extraction_info: Dict[str, Any]
    regulations_data: List[RegulationCategory]
    processing_time: float

class PDFExtractionV2Response(BaseModel):
    success: bool
    regulation_data: List[RegulationCategory] = Field(..., description="추출된 내규 데이터 (벡터 DB 저장용)")
    main_category_name: str = Field(..., description="대분류명")
    extraction_summary: Dict[str, Any] = Field(..., description="추출 요약 정보")

class SaveResponse(BaseModel):
    success: bool
    message: str
    file_path: str
    categories_count: int
    total_faqs: int
    saved_at: str
    file_size_kb: Optional[float] = None

class SaveRequestV2(BaseModel):
    regulation_data: List[RegulationCategory]
    filename: str = Field(..., description="저장할 파일명 (확장자 제외)")

class FileInfo(BaseModel):
    filename: str
    file_path: str
    size_kb: float
    created_at: str
    modified_at: str

class SavedFilesResponse(BaseModel):
    files: List[FileInfo]
    total_count: int
    total_size_kb: float

# 벡터 DB 관련 모델들
class VectorStoreRequest(BaseModel):
    filename: str = Field(..., description="벡터 DB에 저장할 JSON 파일명")
    main_category_name: Optional[str] = Field(None, description="대분류명 (파일명 대신 사용할 경우)")

class VectorStoreDirectRequest(BaseModel):
    regulation_data: List[RegulationCategory]
    main_category_name: str = Field(..., description="대분류명")

class VectorSearchRequest(BaseModel):
    query: str = Field(..., description="검색할 질의", example="휴가 신청 방법")
    top_k: int = Field(10, description="반환할 결과 수", example=10, ge=1, le=50)
    main_category_filter: Optional[str] = Field(None, description="대분류 필터", example="인사")

class VectorSearchResult(BaseModel):
    main_category: str
    sub_category: str
    question: str
    answer: str
    source_file: str
    chunk_type: str
    score: float
    rank: int

class VectorSearchResponse(BaseModel):
    query: str
    results: List[VectorSearchResult]
    count: int
    main_category_filter: Optional[str]
    search_type: str

class VectorStoreResponse(BaseModel):
    success: bool
    message: str
    main_category: str
    total_chunks: int
    chunk_statistics: Dict[str, int]
    vector_db_stats: Dict[str, Any]
    stored_at: str

class VectorStatsResponse(BaseModel):
    total_documents: int
    collection_name: str
    persist_directory: str
    chroma_db_path: str
    chroma_db_size_mb: float
    model_name: str
    main_categories: Dict[str, Any]
    total_main_categories: int
    chunk_statistics: Dict[str, int]
    vector_db_ready: bool

class LLMClient:
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url
        logger.info(f"LLM 클라이언트 초기화: {api_url}")
    
    def create_regulation_extraction_prompt(self, text_content: str, regulation_type: str, extract_mode: str) -> str:
        """내규 JSON 추출용 정교한 시스템 프롬프트 생성"""
        
        current_date = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y년 %m월 %d일")
        
        system_prompt = f"""당신은 회사 내규 문서를 분석하여 정확한 JSON 구조로 변환하는 전문 AI입니다.

🎯 **핵심 임무**: 제공된 내규 텍스트를 분석하여 질문-답변 형태의 구조화된 JSON을 생성하세요.

📋 **출력 형식 (반드시 준수)**:
```json
[
  {{
    "category": "카테고리명 (한글명칭 + 영문명칭)",
    "faqs": [
      {{
        "question": "구체적인 질문",
        "answer": "명확하고 완전한 답변 (조항 번호 포함)"
      }}
    ]
  }}
]
```

🔍 **분석 대상**: {regulation_type} ({extract_mode} 모드)
📅 **분석 일자**: {current_date}

📚 **질문 생성 전략 (모든 경우 대응)**:

1. **기본 정보 질문**:
   - "이 규정의 목적은 무엇인가요?"
   - "적용 범위는 어떻게 되나요?"
   - "용어의 정의는 무엇인가요?"

2. **절차 및 방법 질문**:
   - "~은 어떻게 하나요?" / "~하는 방법은?"
   - "~신청 절차는 어떻게 되나요?"
   - "~처리 기간은 얼마나 걸리나요?"

3. **자격 및 요건 질문**:
   - "~할 수 있는 자격은 무엇인가요?"
   - "~의 요건은 무엇인가요?"
   - "누가 ~할 수 있나요?"

4. **기준 및 조건 질문**:
   - "~의 기준은 무엇인가요?"
   - "어떤 경우에 ~가 가능한가요?"
   - "~의 조건은 무엇인가요?"

5. **권한 및 책임 질문**:
   - "누가 ~을 담당하나요?"
   - "~의 권한은 누구에게 있나요?"
   - "~에 대한 책임은 누가 지나요?"

6. **기간 및 시기 질문**:
   - "~은 언제 하나요?"
   - "~의 기간은 얼마나 되나요?"
   - "~은 몇 년차부터 가능한가요?"

7. **예외 및 특별 상황 질문**:
   - "어떤 경우에 예외가 인정되나요?"
   - "특별한 상황에서는 어떻게 하나요?"
   - "~이 불가능한 경우는 언제인가요?"

8. **계산 및 수치 질문**:
   - "~은 어떻게 계산하나요?"
   - "~의 금액/일수는 얼마인가요?"
   - "~의 비율은 어떻게 되나요?"

9. **변경 및 수정 질문**:
   - "~을 변경하려면 어떻게 하나요?"
   - "~을 수정할 수 있나요?"
   - "~의 개정 절차는 어떻게 되나요?"

10. **위반 및 제재 질문**:
    - "~을 위반하면 어떻게 되나요?"
    - "제재 조치는 무엇인가요?"
    - "벌칙 규정은 어떻게 되나요?"

🎯 **카테고리 분류 가이드**:
- **총칙 (General Provisions)**: 목적, 적용범위, 용어정의
- **인사 (Personnel)**: 채용, 승진, 전보, 퇴직
- **급여 및 수당 (Salary and Allowances)**: 봉급, 수당, 상여금
- **근무 (Work)**: 근무시간, 휴무, 출장
- **휴가 (Leave)**: 연차, 병가, 특별휴가
- **복리후생 (Welfare)**: 각종 복리후생 제도
- **교육 및 연수 (Education and Training)**: 교육, 연수, 자격증
- **평가 및 포상 (Evaluation and Rewards)**: 성과평가, 포상, 징계
- **안전 및 보건 (Safety and Health)**: 안전관리, 보건
- **시행 및 부칙 (Enforcement and Supplementary)**: 시행일, 경과조치

⚡ **품질 기준**:
- **정확성**: 원문 내용을 정확히 반영
- **완전성**: 모든 중요 조항을 빠짐없이 포함
- **명확성**: 이해하기 쉬운 질문과 답변
- **구체성**: 구체적이고 실용적인 질문 생성
- **일관성**: 일관된 형식과 스타일 유지

🚨 **필수 준수사항**:
1. **원문 충실성**: 원문의 조항 번호와 내용을 정확히 인용
2. **JSON 형식**: 올바른 JSON 구조만 출력 (설명 텍스트 금지)
3. **한국어 사용**: 모든 질문과 답변은 한국어로 작성
4. **실용적 질문**: 실제 직원들이 궁금해할 만한 실용적 질문 생성
5. **포괄적 커버리지**: 문서의 모든 중요 내용을 질문-답변으로 변환
5. **빠지지 않은 답변**: 질문-답변 변환시 빠짐없이 모든 조항을 포함

💡 **예시 질문 패턴**:
- "제1조에서 규정하는 목적은 무엇인가요?"
- "21년차 직원의 연차 일수는 며칠인가요?"
- "휴가 신청은 언제까지 해야 하나요?"
- "육아휴직 대상자는 누구인가요?"
- "성과급 지급 기준은 무엇인가요?"

이제 제공된 내규 텍스트를 분석하여 완벽한 JSON을 생성하세요."""

        return system_prompt
    
    def extract_json_from_regulation(self, text_content: str, regulation_type: str = "일반내규", 
                                   extract_mode: str = "comprehensive") -> List[Dict[str, Any]]:
        """내규 텍스트에서 JSON 추출"""
        logger.info(f"내규 JSON 추출 시작: {regulation_type} ({extract_mode} 모드)")
        
        try:
            system_prompt = self.create_regulation_extraction_prompt(text_content, regulation_type, extract_mode)
            
            # 텍스트가 너무 길면 청크 단위로 처리
            max_chunk_size = 8000  # 토큰 제한 고려
            
            if len(text_content) <= max_chunk_size:
                # 단일 처리
                result = self._process_text_chunk(text_content, system_prompt)
            else:
                # 청크 단위 처리 후 병합
                result = self._process_text_in_chunks(text_content, system_prompt, max_chunk_size)
            
            if result:
                logger.info(f"✅ JSON 추출 완료: {len(result)}개 카테고리")
                return result
            else:
                logger.error("❌ JSON 추출 실패: 빈 결과")
                return []
                
        except Exception as e:
            logger.error(f"❌ JSON 추출 중 오류: {e}", exc_info=True)
            return []
    
    def _process_text_chunk(self, text_chunk: str, system_prompt: str) -> List[Dict[str, Any]]:
        """텍스트 청크 처리"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 내규 텍스트를 분석하여 JSON을 생성하세요:\n\n{text_chunk}"}
            ]
            
            response = requests.post(
                self.api_url,
                json={
                    "model": "qwen3-30b-a3b-mlx",
                    "messages": messages,
                    "temperature": 0.1,  # 일관성을 위해 낮은 온도
                    "max_tokens": 40000,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # JSON 추출 및 파싱
                extracted_json = self._extract_and_parse_json(content)
                return extracted_json
            else:
                logger.error(f"❌ LLM API 호출 실패: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ 텍스트 청크 처리 실패: {e}")
            return []
    
    def _process_text_in_chunks(self, text_content: str, system_prompt: str, max_chunk_size: int) -> List[Dict[str, Any]]:
        """긴 텍스트를 청크 단위로 처리"""
        try:
            # 문단 기준으로 분할
            paragraphs = text_content.split('\n\n')
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk + paragraph) <= max_chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            logger.info(f"📄 텍스트를 {len(chunks)}개 청크로 분할")
            
            # 각 청크 처리
            all_results = []
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"🔍 청크 {i}/{len(chunks)} 처리 중...")
                chunk_result = self._process_text_chunk(chunk, system_prompt)
                if chunk_result:
                    all_results.extend(chunk_result)
            
            # 중복 제거 및 병합
            merged_results = self._merge_categories(all_results)
            logger.info(f"✅ 청크 처리 완료: {len(merged_results)}개 카테고리")
            
            return merged_results
            
        except Exception as e:
            logger.error(f"❌ 청크 처리 실패: {e}")
            return []
    
    def _extract_and_parse_json(self, content: str) -> List[Dict[str, Any]]:
        """응답에서 JSON 추출 및 파싱"""
        try:
            # JSON 블록 찾기
            json_patterns = [
                r'```json\s*(\[.*?\])\s*```',
                r'```\s*(\[.*?\])\s*```',
                r'(\[.*?\])',
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
                r'(\{.*?\})'
            ]
            
            extracted_json = None
            
            for pattern in json_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    json_str = matches[0]
                    try:
                        extracted_json = json.loads(json_str)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if extracted_json is None:
                # 직접 JSON 파싱 시도
                try:
                    extracted_json = json.loads(content)
                except json.JSONDecodeError:
                    logger.error("❌ JSON 파싱 실패")
                    return []
            
            # 형식 검증 및 정규화
            if isinstance(extracted_json, dict) and 'category' in extracted_json:
                extracted_json = [extracted_json]
            
            if isinstance(extracted_json, list):
                validated_data = []
                for item in extracted_json:
                    if isinstance(item, dict) and 'category' in item and 'faqs' in item:
                        validated_data.append(item)
                
                return validated_data
            
            return []
            
        except Exception as e:
            logger.error(f"❌ JSON 추출 및 파싱 실패: {e}")
            return []
    
    def _merge_categories(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """카테고리 중복 제거 및 병합"""
        try:
            category_map = {}
            
            for item in all_results:
                category_name = item.get('category', '')
                if category_name in category_map:
                    # 기존 카테고리에 FAQ 추가
                    existing_questions = {faq['question'] for faq in category_map[category_name]['faqs']}
                    for faq in item.get('faqs', []):
                        if faq['question'] not in existing_questions:
                            category_map[category_name]['faqs'].append(faq)
                            existing_questions.add(faq['question'])
                else:
                    category_map[category_name] = item
            
            return list(category_map.values())
            
        except Exception as e:
            logger.error(f"❌ 카테고리 병합 실패: {e}")
            return all_results

class PDFProcessor:
    """PDF 처리 클래스"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_content: bytes) -> str:
        """PDF에서 텍스트 추출"""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += f"\n\n=== 페이지 {page_num + 1} ===\n{text}"
            
            doc.close()
            
            # 텍스트 정리
            cleaned_text = PDFProcessor._clean_extracted_text(full_text)
            logger.info(f"📄 PDF 텍스트 추출 완료: {len(cleaned_text)}자")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"❌ PDF 텍스트 추출 실패: {e}")
            raise HTTPException(status_code=400, detail=f"PDF 처리 실패: {str(e)}")
    
    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """추출된 텍스트 정리"""
        try:
            # 불필요한 공백 제거
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            
            # 페이지 구분자 정리
            text = re.sub(r'=== 페이지 \d+ ===\n', '', text)
            
            # 빈 줄 정리
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"❌ 텍스트 정리 실패: {e}")
            return text

class VectorDBSystem:
    """벡터 DB 시스템 클래스 (RAG 서버 호환)"""
    
    def __init__(self, model_name: str = "nlpai-lab/KURE-v1", persist_directory: str = "./chroma_db"):
        """벡터 DB 시스템 초기화 (RAG 서버와 동일한 방식)"""
        self.model_name = model_name
        self.persist_directory = persist_directory
        
        # ChromaDB 저장 디렉토리 생성
        os.makedirs(self.persist_directory, exist_ok=True)
        logger.info(f"📁 ChromaDB 저장 경로: {os.path.abspath(self.persist_directory)}")
        
        # 모델 경로 설정 (RAG 서버와 동일한 방식)
        model_path = os.path.join("./models", model_name.replace("/", "-"))
        
        # 모델 로드
        if not os.path.exists(model_path):
            logger.info(f"📦 모델 다운로드: '{model_name}'")
            try:
                os.makedirs("./models", exist_ok=True)
                model = SentenceTransformer(model_name)
                model.save(model_path)
                logger.info(f"✅ 모델 저장 완료: {model_path}")
            except Exception as e:
                logger.error(f"❌ 모델 다운로드 실패: {e}", exc_info=True)
                raise
        else:
            logger.info(f"🔄 기존 모델 로드: {model_path}")

        self.model = SentenceTransformer(model_path)
        
        # ChromaDB 초기화 (RAG 서버와 동일한 방식)
        logger.info(f"🔍 ChromaDB 초기화 중: {self.persist_directory}")
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            self.collection = self.chroma_client.get_collection(name="company_regulations")
            logger.info(f"📁 기존 company_regulations 컬렉션 로드 완료 (경로: {self.persist_directory})")
        except Exception as get_error:
            logger.info(f"📁 새 컬렉션 생성 중... (경로: {self.persist_directory})")
            try:
                self.collection = self.chroma_client.create_collection(
                    name="company_regulations",
                    metadata={
                        "description": "회사 내규 벡터 검색 컬렉션 (RAG 서버 호환)",
                        "persist_directory": self.persist_directory,
                        "created_at": datetime.now(pytz.timezone('Asia/Seoul')).isoformat()
                    }
                )
                logger.info(f"📁 새 컬렉션 생성 완료 (경로: {self.persist_directory})")
            except Exception as create_error:
                import time
                temp_name = f"company_regulations_{int(time.time())}"
                self.collection = self.chroma_client.create_collection(
                    name=temp_name,
                    metadata={
                        "description": "임시 컬렉션",
                        "persist_directory": self.persist_directory,
                        "created_at": datetime.now(pytz.timezone('Asia/Seoul')).isoformat()
                    }
                )
                logger.info(f"📁 임시 컬렉션 '{temp_name}' 생성 완료 (경로: {self.persist_directory})")

        logger.info(f"✅ 벡터 DB 시스템 초기화 완료 (ChromaDB: {self.persist_directory}, RAG 서버 호환)")
    
    def store_regulation_data(self, regulation_data: List[Dict[str, Any]], main_category_name: str) -> Dict[str, Any]:
        """내규 데이터를 벡터 DB에 저장 (RAG 서버와 완전 동일한 방식)"""
        logger.info(f"🔍 벡터 DB 저장 시작 (RAG 호환): {main_category_name} (ChromaDB: {self.persist_directory})")
        
        try:
            # RAG 서버의 regulations_data 구조로 변환
            regulations_data = []
            
            for category_section in regulation_data:
                sub_category = category_section['category']
                
                for faq in category_section['faqs']:
                    question = faq['question']
                    answer = faq['answer']
                    
                    # RAG 서버와 동일한 청킹 전략
                    # 1. 기본 Q&A 단위 저장
                    base_id = str(uuid.uuid4())
                    regulation_item = {
                        'id': base_id,
                        'main_category': main_category_name,
                        'sub_category': sub_category,
                        'question': question,
                        'answer': answer,
                        'text': f"{question} {answer}",
                        'source_file': f"{main_category_name}.json",
                        'chunk_type': 'qa_full',
                        'chunk_id': 0
                    }
                    regulations_data.append(regulation_item)
                    
                    # 2. 향상된 청킹 (RAG 서버와 동일)
                    question_item = {
                        'id': f"{base_id}_q",
                        'main_category': main_category_name,
                        'sub_category': sub_category,
                        'question': question,
                        'answer': answer,
                        'text': f"질문: {question}",
                        'source_file': f"{main_category_name}.json",
                        'chunk_type': 'question_focused',
                        'chunk_id': 1
                    }
                    regulations_data.append(question_item)
                    
                    answer_item = {
                        'id': f"{base_id}_a",
                        'main_category': main_category_name,
                        'sub_category': sub_category,
                        'question': question,
                        'answer': answer,
                        'text': f"답변: {answer} (관련 질문: {question})",
                        'source_file': f"{main_category_name}.json",
                        'chunk_type': 'answer_focused',
                        'chunk_id': 2
                    }
                    regulations_data.append(answer_item)
                    
                    # 3. 긴 답변의 경우 문장 분할 (RAG 서버와 동일)
                    if len(answer) > 200:
                        sentences = re.split(r'[.!?]\s+', answer)
                        for i, sentence in enumerate(sentences):
                            if len(sentence.strip()) > 20:
                                sentence_item = {
                                    'id': f"{base_id}_s{i}",
                                    'main_category': main_category_name,
                                    'sub_category': sub_category,
                                    'question': question,
                                    'answer': answer,
                                    'text': f"{sentence.strip()} (출처: {question})",
                                    'source_file': f"{main_category_name}.json",
                                    'chunk_type': 'sentence_level',
                                    'chunk_id': 10 + i
                                }
                                regulations_data.append(sentence_item)
            
            # 청킹 통계 계산 (RAG 서버와 동일)
            chunk_statistics = {}
            for item in regulations_data:
                chunk_type = item.get('chunk_type', 'unknown')
                chunk_statistics[chunk_type] = chunk_statistics.get(chunk_type, 0) + 1
            
            # ChromaDB에 저장 (RAG 서버와 동일한 방식)
            logger.info(f"⚙️ 임베딩 및 저장: {len(regulations_data)}개 청크")
            
            texts = [item['text'] for item in regulations_data]
            metadatas = [
                {
                    'main_category': item['main_category'],
                    'sub_category': item['sub_category'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'source_file': item['source_file'],
                    'chunk_type': item.get('chunk_type', 'qa_full'),
                    'chunk_id': item.get('chunk_id', 0)
                } 
                for item in regulations_data
            ]
            ids = [item['id'] for item in regulations_data]
            
            # 배치 처리 (RAG 서버와 동일)
            batch_size = 500
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                try:
                    embeddings = self.model.encode(batch_texts, show_progress_bar=False).tolist()
                    
                    self.collection.add(
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        embeddings=embeddings,
                        ids=batch_ids
                    )
                    
                    logger.info(f"📦 배치 {i//batch_size + 1}/{total_batches} 완료")
                    
                except Exception as batch_error:
                    logger.error(f"❌ 배치 처리 실패: {batch_error}")
                    continue
            
            final_count = self.collection.count()
            logger.info(f"✅ 벡터 DB 저장 완료 (RAG 호환): {len(regulations_data)}개 청크 저장 (ChromaDB: {self.persist_directory})")
            
            return {
                "total_chunks": len(regulations_data),
                "chunk_statistics": chunk_statistics,
                "vector_db_total": final_count,
                "batch_count": total_batches
            }
            
        except Exception as e:
            logger.error(f"❌ 벡터 DB 저장 실패: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"벡터 DB 저장 실패: {str(e)}")
    
    def search(self, query: str, top_k: int = 10, main_category_filter: str = None) -> List[Dict[str, Any]]:
        """벡터 DB에서 검색 (RAG 서버와 동일한 방식)"""
        logger.info(f"🔍 벡터 검색 실행: '{query[:50]}...', top_k={top_k} (ChromaDB: {self.persist_directory})")
        
        try:
            if self.collection.count() == 0:
                logger.warning("⚠️ 빈 인덱스")
                return []
            
            # 필터 조건
            where_condition = None
            if main_category_filter:
                where_condition = {"main_category": main_category_filter}
            
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query]).tolist()
            
            # 검색 수행
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['documents', 'metadatas', 'distances'],
                where=where_condition
            )
            
            # 결과 처리 (RAG 서버와 동일)
            search_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    # 거리를 유사도 점수로 변환
                    similarity_score = max(0, 1 - (distance / 2))
                    
                    search_results.append({
                        'main_category': metadata['main_category'],
                        'sub_category': metadata['sub_category'],
                        'question': metadata['question'],
                        'answer': metadata['answer'],
                        'source_file': metadata['source_file'],
                        'chunk_type': metadata.get('chunk_type', 'qa_full'),
                        'score': similarity_score,
                        'rank': i + 1
                    })
            
            logger.info(f"✅ 벡터 검색 완료: {len(search_results)}개 결과 (ChromaDB: {self.persist_directory})")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ 벡터 검색 실패: {e}", exc_info=True)
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """벡터 DB 통계 정보 반환 (RAG 서버와 동일)"""
        try:
            count = self.collection.count()
            
            # ChromaDB 디렉토리 정보
            chroma_db_path = os.path.abspath(self.persist_directory)
            chroma_db_size = 0
            if os.path.exists(self.persist_directory):
                for root, dirs, files in os.walk(self.persist_directory):
                    for file in files:
                        chroma_db_size += os.path.getsize(os.path.join(root, file))
            
            # 메타데이터 분석을 위해 샘플 데이터 가져오기
            if count > 0:
                sample_results = self.collection.get(limit=min(1000, count))
                main_categories = {}
                chunk_statistics = {}
                
                if sample_results and 'metadatas' in sample_results:
                    for metadata in sample_results['metadatas']:
                        main_cat = metadata.get('main_category', 'Unknown')
                        chunk_type = metadata.get('chunk_type', 'unknown')
                        
                        if main_cat not in main_categories:
                            main_categories[main_cat] = {
                                'source_file': metadata.get('source_file', 'Unknown'),
                                'count': 0
                            }
                        main_categories[main_cat]['count'] += 1
                        
                        chunk_statistics[chunk_type] = chunk_statistics.get(chunk_type, 0) + 1
            else:
                main_categories = {}
                chunk_statistics = {}
            
            return {
                'total_documents': count,
                'collection_name': self.collection.name,
                'persist_directory': self.persist_directory,
                'chroma_db_path': chroma_db_path,
                'chroma_db_size_mb': round(chroma_db_size / (1024 * 1024), 2),
                'model_name': self.model_name,
                'vector_db_ready': count > 0,
                'main_categories': main_categories,
                'total_main_categories': len(main_categories),
                'chunk_statistics': chunk_statistics
            }
        except Exception as e:
            logger.error(f"❌ 통계 조회 실패: {e}")
            return {
                'total_documents': 0, 
                'vector_db_ready': False,
                'persist_directory': self.persist_directory,
                'chroma_db_path': os.path.abspath(self.persist_directory),
                'error': str(e)
            }

# 전역 객체
llm_client = LLMClient()
vector_db_system = None

def initialize_vector_db():
    """벡터 DB 시스템 초기화"""
    global vector_db_system
    
    try:
        if vector_db_system is None:
            logger.info("🔍 벡터 DB 시스템 초기화 중... (저장 위치: ./chroma_db, RAG 서버 호환)")
            vector_db_system = VectorDBSystem(persist_directory="./chroma_db")
            logger.info("✅ 벡터 DB 시스템 초기화 완료 (ChromaDB: ./chroma_db, RAG 서버 호환)")
        return True
    except Exception as e:
        logger.error(f"❌ 벡터 DB 시스템 초기화 실패: {e}")
        return False

@app.post("/api/extract_regulation_json", response_model=PDFExtractionResponse, 
         summary="내규 PDF JSON 추출", 
         description="업로드된 내규 PDF를 분석하여 구조화된 JSON 형태로 변환합니다.")
async def extract_regulation_json(
    file: UploadFile = File(..., description="내규 PDF 파일"),
    regulation_type: str = Form("일반내규", description="내규 유형 (예: 직제규정, 인사규정, 급여규정)"),
    extract_mode: str = Form("comprehensive", description="추출 모드 (comprehensive: 포괄적, focused: 핵심만)")
):
    """내규 PDF에서 JSON 추출"""
    start_time = datetime.now()
    
    try:
        # 파일 검증
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
        
        logger.info(f"📄 내규 PDF 처리 시작: {file.filename} ({regulation_type}, {extract_mode})")
        
        # PDF 내용 읽기
        pdf_content = await file.read()
        
        # 텍스트 추출
        extracted_text = PDFProcessor.extract_text_from_pdf(pdf_content)
        
        if len(extracted_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="PDF에서 충분한 텍스트를 추출할 수 없습니다.")
        
        # LLM으로 JSON 추출
        regulations_data = llm_client.extract_json_from_regulation(
            extracted_text, 
            regulation_type, 
            extract_mode
        )
        
        if not regulations_data:
            raise HTTPException(status_code=500, detail="내규 JSON 추출에 실패했습니다.")
        
        # 통계 계산
        total_faqs = sum(len(category.get('faqs', [])) for category in regulations_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 추출 정보
        extraction_info = {
            "original_filename": file.filename,
            "file_size_mb": round(len(pdf_content) / (1024 * 1024), 2),
            "extracted_text_length": len(extracted_text),
            "processing_time_seconds": round(processing_time, 2),
            "extraction_timestamp": datetime.now(pytz.timezone('Asia/Seoul')).isoformat(),
            "llm_model": "qwen3-30b-a3b-mlx",
            "extraction_strategy": {
                "mode": extract_mode,
                "question_generation": "다각도 질문 생성 (10가지 패턴)",
                "category_classification": "의미 기반 자동 분류",
                "quality_assurance": "원문 충실성 + 실용성 검증"
            }
        }
        
        logger.info(f"✅ 내규 JSON 추출 완료: {len(regulations_data)}개 카테고리, {total_faqs}개 FAQ")
        
        return PDFExtractionResponse(
            success=True,
            regulation_type=regulation_type,
            categories_count=len(regulations_data),
            total_faqs=total_faqs,
            extraction_info=extraction_info,
            regulations_data=[RegulationCategory(**data) for data in regulations_data],
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 내규 JSON 추출 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"내규 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/extract_regulation_json_v2", response_model=PDFExtractionV2Response, 
         summary="내규 PDF JSON 추출 (벡터 DB 저장용)", 
         description="업로드된 내규 PDF를 분석하여 벡터 DB에 바로 저장할 수 있는 형태로 JSON을 변환합니다.")
async def extract_regulation_json_v2(
    file: UploadFile = File(..., description="내규 PDF 파일"),
    regulation_type: str = Form("일반내규", description="내규 유형 (예: 직제규정, 인사규정, 급여규정)"),
    extract_mode: str = Form("comprehensive", description="추출 모드 (comprehensive: 포괄적, focused: 핵심만)")
):
    """내규 PDF에서 JSON 추출 (벡터 DB 저장용 형태로 반환)"""
    start_time = datetime.now()
    
    try:
        # 파일 검증
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
        
        logger.info(f"📄 내규 PDF 처리 시작 (v2): {file.filename} ({regulation_type}, {extract_mode})")
        
        # PDF 내용 읽기
        pdf_content = await file.read()
        
        # 텍스트 추출
        extracted_text = PDFProcessor.extract_text_from_pdf(pdf_content)
        
        if len(extracted_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="PDF에서 충분한 텍스트를 추출할 수 없습니다.")
        
        # LLM으로 JSON 추출
        regulations_data = llm_client.extract_json_from_regulation(
            extracted_text, 
            regulation_type, 
            extract_mode
        )
        
        if not regulations_data:
            raise HTTPException(status_code=500, detail="내규 JSON 추출에 실패했습니다.")
        
        # 통계 계산
        total_faqs = sum(len(category.get('faqs', [])) for category in regulations_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 대분류명 결정 (파일명 기반 또는 내규 유형 기반)
        main_category_name = regulation_type
        if main_category_name == "일반내규":
            # 파일명에서 추출 시도
            filename_without_ext = os.path.splitext(file.filename)[0]
            # 한글, 영문, 숫자만 남기고 정리
            cleaned_filename = re.sub(r'[^\w가-힣]', '', filename_without_ext)
            if cleaned_filename:
                main_category_name = cleaned_filename
        
        # RegulationCategory 객체로 변환
        regulation_categories = [RegulationCategory(**data) for data in regulations_data]
        
        # 추출 요약 정보
        extraction_summary = {
            "original_filename": file.filename,
            "main_category_name": main_category_name,
            "regulation_type": regulation_type,
            "extract_mode": extract_mode,
            "categories_count": len(regulations_data),
            "total_faqs": total_faqs,
            "file_size_mb": round(len(pdf_content) / (1024 * 1024), 2),
            "extracted_text_length": len(extracted_text),
            "processing_time_seconds": round(processing_time, 2),
            "extraction_timestamp": datetime.now(pytz.timezone('Asia/Seoul')).isoformat(),
            "llm_model": "qwen3-30b-a3b-mlx",
            "ready_for_vector_storage": True,
            "vector_db_compatibility": "RAG 서버 호환 형식",
            "usage_note": "이 데이터는 /api/store_extracted_json_to_vector API에 바로 사용할 수 있습니다."
        }
        
        logger.info(f"✅ 내규 JSON 추출 완료 (v2): {len(regulations_data)}개 카테고리, {total_faqs}개 FAQ → 대분류: {main_category_name}")
        
        return PDFExtractionV2Response(
            success=True,
            regulation_data=regulation_categories,
            main_category_name=main_category_name,
            extraction_summary=extraction_summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 내규 JSON 추출 실패 (v2): {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"내규 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/save_extracted_json_v2", 
         response_model=SaveResponse,
         summary="추출된 JSON 저장 (JSON Body)", 
         description="JSON body로 내규 데이터를 받아 파일로 저장합니다. (권장 방식)")
async def save_extracted_json_v2(request_data: SaveRequestV2):
    """JSON body로 추출된 데이터를 파일로 저장"""
    start_time = datetime.now()
    logger.info(f"📁 JSON 저장 요청 받음 (v2): filename={request_data.filename}")
    
    try:
        # 데이터 검증은 Pydantic 모델에서 자동으로 처리됨
        regulation_data = request_data.regulation_data
        filename = request_data.filename
        
        # 저장할 디렉토리 확인
        save_directory = "./extracted_regulations"
        os.makedirs(save_directory, exist_ok=True)
        
        # 파일명 안전하게 처리
        safe_filename = re.sub(r'[^\w\-_가-힣]', '_', filename)
        if not safe_filename:
            safe_filename = f"regulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        file_path = os.path.join(save_directory, f"{safe_filename}.json")
        
        # 파일 중복 체크 및 번호 추가
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_path)
            file_path = f"{name}_{counter}{ext}"
            counter += 1
        
        # JSON 데이터 변환
        output_data = [category.dict() for category in regulation_data]
        
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # 파일 크기 확인
        file_size_kb = round(os.path.getsize(file_path) / 1024, 2)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ JSON 파일 저장 완료 (v2): {file_path} ({file_size_kb}KB, {processing_time:.2f}초)")
        
        return SaveResponse(
            success=True,
            message="JSON 파일이 성공적으로 저장되었습니다.",
            file_path=file_path,
            categories_count=len(output_data),
            total_faqs=sum(len(cat['faqs']) for cat in output_data),
            saved_at=datetime.now(pytz.timezone('Asia/Seoul')).isoformat(),
            file_size_kb=file_size_kb
        )
        
    except Exception as e:
        logger.error(f"❌ JSON 저장 실패 (v2): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {str(e)}")

@app.post("/api/store_json_to_vector", 
         response_model=VectorStoreResponse,
         summary="JSON 파일을 벡터 DB에 저장 (RAG 서버 호환)", 
         description="저장된 JSON 파일을 읽어와서 RAG 서버와 동일한 형식으로 벡터 데이터베이스에 저장합니다.")
async def store_json_to_vector(request: VectorStoreRequest):
    """저장된 JSON 파일을 벡터 DB에 저장 (RAG 서버 호환)"""
    try:
        # 벡터 DB 시스템 초기화
        if not initialize_vector_db():
            raise HTTPException(status_code=503, detail="벡터 DB 시스템 초기화 실패")
        
        # 파일 경로 생성
        filename = request.filename
        if not filename.endswith('.json'):
            filename += '.json'
        
        file_path = os.path.join("./extracted_regulations", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {filename}")
        
        # JSON 파일 읽기
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                regulation_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON 파일 읽기 실패: {str(e)}")
        
        # 대분류명 결정
        main_category_name = request.main_category_name or os.path.splitext(filename)[0]
        
        # 벡터 DB에 저장 (RAG 서버와 동일한 방식)
        store_result = vector_db_system.store_regulation_data(regulation_data, main_category_name)
        
        logger.info(f"✅ 벡터 DB 저장 완료 (RAG 호환): {filename} → {main_category_name}")
        
        return VectorStoreResponse(
            success=True,
            message=f"JSON 파일 '{filename}'이 RAG 서버 호환 형식으로 벡터 DB에 성공적으로 저장되었습니다.",
            main_category=main_category_name,
            total_chunks=store_result["total_chunks"],
            chunk_statistics=store_result["chunk_statistics"],
            vector_db_stats={
                "total_documents": store_result["vector_db_total"],
                "batch_count": store_result["batch_count"],
                "rag_compatibility": "완전 호환"
            },
            stored_at=datetime.now(pytz.timezone('Asia/Seoul')).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 벡터 DB 저장 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"벡터 DB 저장 실패: {str(e)}")

@app.post("/api/store_extracted_json_to_vector", 
         response_model=VectorStoreResponse,
         summary="추출된 JSON을 직접 벡터 DB에 저장 (RAG 서버 호환)", 
         description="추출된 JSON 데이터를 파일 저장 없이 RAG 서버와 동일한 형식으로 직접 벡터 데이터베이스에 저장합니다.")
async def store_extracted_json_to_vector(request: VectorStoreDirectRequest):
    """추출된 JSON 데이터를 직접 벡터 DB에 저장 (RAG 서버 호환)"""
    try:
        # 벡터 DB 시스템 초기화
        if not initialize_vector_db():
            raise HTTPException(status_code=503, detail="벡터 DB 시스템 초기화 실패")
        
        # JSON 데이터 변환
        regulation_data = [category.dict() for category in request.regulation_data]
        
        # 벡터 DB에 저장 (RAG 서버와 동일한 방식)
        store_result = vector_db_system.store_regulation_data(regulation_data, request.main_category_name)
        
        logger.info(f"✅ 벡터 DB 직접 저장 완료 (RAG 호환): {request.main_category_name}")
        
        return VectorStoreResponse(
            success=True,
            message=f"JSON 데이터가 RAG 서버 호환 형식으로 벡터 DB에 성공적으로 저장되었습니다. (대분류: {request.main_category_name})",
            main_category=request.main_category_name,
            total_chunks=store_result["total_chunks"],
            chunk_statistics=store_result["chunk_statistics"],
            vector_db_stats={
                "total_documents": store_result["vector_db_total"],
                "batch_count": store_result["batch_count"],
                "rag_compatibility": "완전 호환",
                "chunking_strategy": "qa_full + question_focused + answer_focused + sentence_level"
            },
            stored_at=datetime.now(pytz.timezone('Asia/Seoul')).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 벡터 DB 직접 저장 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"벡터 DB 직접 저장 실패: {str(e)}")

@app.post("/api/vector_search", 
         response_model=VectorSearchResponse,
         summary="벡터 DB 검색 (RAG 서버 호환)", 
         description="RAG 서버와 동일한 방식으로 벡터 데이터베이스에서 의미 기반 검색을 수행합니다.")
async def vector_search(request: VectorSearchRequest):
    """벡터 DB에서 검색 (RAG 서버 호환)"""
    try:
        # 벡터 DB 시스템 초기화 시도
        if vector_db_system is None:
            logger.info("🔍 벡터 DB 시스템 자동 초기화 시도...")
            if not initialize_vector_db():
                raise HTTPException(
                    status_code=503, 
                    detail={
                        "error": "벡터 DB 시스템 초기화 실패",
                        "message": "벡터 DB 시스템을 초기화할 수 없습니다. 시스템 로그를 확인해주세요.",
                        "suggestions": [
                            "모델 다운로드가 필요할 수 있습니다",
                            "ChromaDB 저장 공간을 확인해주세요",
                            "네트워크 연결을 확인해주세요"
                        ]
                    }
                )
        
        # 벡터 DB가 비어있는지 확인
        stats = vector_db_system.get_stats()
        if stats['total_documents'] == 0:
            return VectorSearchResponse(
                query=request.query,
                results=[],
                count=0,
                main_category_filter=request.main_category_filter,
                search_type="벡터 기반 의미 검색 (RAG 호환, 데이터 없음)"
            )
        
        # 검색 수행 (RAG 서버와 동일한 방식)
        results = vector_db_system.search(
            query=request.query,
            top_k=request.top_k,
            main_category_filter=request.main_category_filter
        )
        
        # 결과 변환
        search_results = [VectorSearchResult(**result) for result in results]
        
        return VectorSearchResponse(
            query=request.query,
            results=search_results,
            count=len(results),
            main_category_filter=request.main_category_filter,
            search_type="벡터 기반 의미 검색 (RAG 서버 호환)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 벡터 검색 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"벡터 검색 실패: {str(e)}")

@app.get("/api/vector_stats", 
         response_model=VectorStatsResponse,
         summary="벡터 DB 통계 (RAG 서버 호환)", 
         description="RAG 서버와 동일한 형식으로 벡터 데이터베이스의 상태와 통계 정보를 확인합니다. (저장 위치: ./chroma_db)")
async def get_vector_stats():
    """벡터 DB 통계 정보 (RAG 서버 호환)"""
    try:
        if vector_db_system is None:
            # 초기화 시도
            if not initialize_vector_db():
                raise HTTPException(status_code=503, detail="벡터 DB 시스템이 초기화되지 않았습니다.")
        
        stats = vector_db_system.get_stats()
        
        return VectorStatsResponse(
            total_documents=stats['total_documents'],
            collection_name=stats['collection_name'],
            persist_directory=stats['persist_directory'],
            chroma_db_path=stats.get('chroma_db_path', './chroma_db'),
            chroma_db_size_mb=stats.get('chroma_db_size_mb', 0.0),
            model_name=stats.get('model_name', 'nlpai-lab/KURE-v1'),
            main_categories=stats['main_categories'],
            total_main_categories=stats['total_main_categories'],
            chunk_statistics=stats['chunk_statistics'],
            vector_db_ready=stats['vector_db_ready']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 벡터 DB 통계 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"벡터 DB 통계 조회 실패: {str(e)}")

@app.get("/api/saved_files", 
         response_model=SavedFilesResponse,
         summary="저장된 파일 목록 조회")
async def get_saved_files():
    """저장된 내규 JSON 파일 목록 반환"""
    try:
        save_directory = "./extracted_regulations"
        if not os.path.exists(save_directory):
            return SavedFilesResponse(files=[], total_count=0, total_size_kb=0.0)
        
        files = []
        for filename in os.listdir(save_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(save_directory, filename)
                stat = os.stat(file_path)
                
                files.append(FileInfo(
                    filename=filename,
                    file_path=file_path,
                    size_kb=round(stat.st_size / 1024, 2),
                    created_at=datetime.fromtimestamp(stat.st_ctime, pytz.timezone('Asia/Seoul')).isoformat(),
                    modified_at=datetime.fromtimestamp(stat.st_mtime, pytz.timezone('Asia/Seoul')).isoformat()
                ))
        
        # 수정 시간 기준 내림차순 정렬
        files.sort(key=lambda x: x.modified_at, reverse=True)
        
        return SavedFilesResponse(
            files=files,
            total_count=len(files),
            total_size_kb=round(sum(f.size_kb for f in files), 2)
        )
        
    except Exception as e:
        logger.error(f"❌ 파일 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 목록 조회 실패: {str(e)}")

@app.get("/api/download_file/{filename}", summary="저장된 파일 다운로드")
async def download_file(filename: str):
    """저장된 JSON 파일 다운로드"""
    try:
        # 파일명 보안 검증
        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith('.json'):
            safe_filename += '.json'
        
        file_path = os.path.join("./extracted_regulations", safe_filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return JSONResponse(
            content=json.loads(content),
            headers={
                "Content-Disposition": f"attachment; filename={safe_filename}",
                "Content-Type": "application/json; charset=utf-8"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 파일 다운로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 다운로드 실패: {str(e)}")

@app.delete("/api/delete_file/{filename}", summary="저장된 파일 삭제")
async def delete_file(filename: str):
    """저장된 JSON 파일 삭제"""
    try:
        # 파일명 보안 검증
        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith('.json'):
            safe_filename += '.json'
        
        file_path = os.path.join("./extracted_regulations", safe_filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        # 파일 삭제
        os.remove(file_path)
        
        logger.info(f"🗑️ 파일 삭제 완료: {file_path}")
        
        return {
            "success": True,
            "message": f"파일 '{safe_filename}'이 성공적으로 삭제되었습니다.",
            "deleted_file": safe_filename,
            "deleted_at": datetime.now(pytz.timezone('Asia/Seoul')).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 파일 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 삭제 실패: {str(e)}")

@app.get("/api/export_vector_to_json/{main_category}", 
         summary="벡터 DB 데이터를 JSON으로 내보내기 (RAG 서버 호환)", 
         description="벡터 DB에 저장된 특정 대분류 데이터를 RAG 서버가 읽을 수 있는 JSON 파일로 내보냅니다.")
async def export_vector_to_json(main_category: str):
    """벡터 DB의 데이터를 JSON 파일로 내보내기 (RAG 서버 호환)"""
    try:
        if vector_db_system is None:
            if not initialize_vector_db():
                raise HTTPException(status_code=503, detail="벡터 DB 시스템이 초기화되지 않았습니다.")
        
        # 벡터 DB에서 특정 대분류의 모든 데이터 가져오기
        all_results = vector_db_system.collection.get(
            where={"main_category": main_category},
            include=['documents', 'metadatas']
        )
        
        if not all_results or not all_results.get('metadatas'):
            raise HTTPException(status_code=404, detail=f"대분류 '{main_category}'의 데이터를 찾을 수 없습니다.")
        
        # 메타데이터를 카테고리별로 그룹화 (RAG 서버 형식에 맞춤)
        categories_dict = {}
        
        for metadata in all_results['metadatas']:
            # qa_full 청크만 사용 (중복 방지)
            if metadata.get('chunk_type') != 'qa_full':
                continue
                
            sub_category = metadata['sub_category']
            question = metadata['question']
            answer = metadata['answer']
            
            if sub_category not in categories_dict:
                categories_dict[sub_category] = {
                    "category": sub_category,
                    "faqs": []
                }
            
            categories_dict[sub_category]["faqs"].append({
                "question": question,
                "answer": answer
            })
        
        # 리스트 형태로 변환 (RAG 서버가 기대하는 형식)
        exported_data = list(categories_dict.values())
        
        # data_json 디렉토리에 저장 (RAG 서버 호환)
        data_json_dir = "./data_json"
        os.makedirs(data_json_dir, exist_ok=True)
        
        output_filename = f"{main_category}.json"
        output_path = os.path.join(data_json_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(exported_data, f, ensure_ascii=False, indent=2)
        
        file_size_kb = round(os.path.getsize(output_path) / 1024, 2)
        
        logger.info(f"✅ 벡터 DB 데이터 내보내기 완료 (RAG 호환): {output_path}")
        
        return {
            "success": True,
            "message": f"'{main_category}' 데이터를 RAG 서버 호환 JSON 파일로 내보냈습니다.",
            "main_category": main_category,
            "output_file": output_filename,
            "output_path": output_path,
            "categories_count": len(exported_data),
            "total_faqs": sum(len(cat["faqs"]) for cat in exported_data),
            "file_size_kb": file_size_kb,
            "exported_at": datetime.now(pytz.timezone('Asia/Seoul')).isoformat(),
            "rag_compatibility": "완전 호환",
            "usage_note": "포트 5000의 RAG 서버에서 /api/rebuild_index를 호출하여 데이터를 로드할 수 있습니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 벡터 DB 데이터 내보내기 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"데이터 내보내기 실패: {str(e)}")

@app.get("/api/export_all_vector_to_json", 
         summary="벡터 DB의 모든 데이터를 JSON으로 내보내기 (RAG 서버 호환)", 
         description="벡터 DB에 저장된 모든 대분류 데이터를 RAG 서버가 읽을 수 있는 JSON 파일들로 내보냅니다.")
async def export_all_vector_to_json():
    """벡터 DB의 모든 데이터를 JSON 파일들로 내보내기 (RAG 서버 호환)"""
    try:
        if vector_db_system is None:
            if not initialize_vector_db():
                raise HTTPException(status_code=503, detail="벡터 DB 시스템이 초기화되지 않았습니다.")
        
        # 벡터 DB 통계에서 모든 대분류 가져오기
        stats = vector_db_system.get_stats()
        main_categories = list(stats['main_categories'].keys())
        
        if not main_categories:
            raise HTTPException(status_code=404, detail="벡터 DB에 데이터가 없습니다.")
        
        # data_json 디렉토리 준비 (RAG 서버 호환)
        data_json_dir = "./data_json"
        os.makedirs(data_json_dir, exist_ok=True)
        
        exported_files = []
        total_categories = 0
        total_faqs = 0
        
        for main_category in main_categories:
            try:
                # 각 대분류별로 데이터 추출
                all_results = vector_db_system.collection.get(
                    where={"main_category": main_category},
                    include=['documents', 'metadatas']
                )
                
                if not all_results or not all_results.get('metadatas'):
                    continue
                
                # 메타데이터를 카테고리별로 그룹화 (RAG 서버 형식에 맞춤)
                categories_dict = {}
                
                for metadata in all_results['metadatas']:
                    # qa_full 청크만 사용 (중복 방지)
                    if metadata.get('chunk_type') != 'qa_full':
                        continue
                        
                    sub_category = metadata['sub_category']
                    question = metadata['question']
                    answer = metadata['answer']
                    
                    if sub_category not in categories_dict:
                        categories_dict[sub_category] = {
                            "category": sub_category,
                            "faqs": []
                        }
                    
                    categories_dict[sub_category]["faqs"].append({
                        "question": question,
                        "answer": answer
                    })
                
                # 리스트 형태로 변환 (RAG 서버가 기대하는 형식)
                exported_data = list(categories_dict.values())
                
                # 파일 저장
                output_filename = f"{main_category}.json"
                output_path = os.path.join(data_json_dir, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(exported_data, f, ensure_ascii=False, indent=2)
                
                file_size_kb = round(os.path.getsize(output_path) / 1024, 2)
                faqs_count = sum(len(cat["faqs"]) for cat in exported_data)
                
                exported_files.append({
                    "main_category": main_category,
                    "filename": output_filename,
                    "categories_count": len(exported_data),
                    "faqs_count": faqs_count,
                    "file_size_kb": file_size_kb
                })
                
                total_categories += len(exported_data)
                total_faqs += faqs_count
                
                logger.info(f"✅ 내보내기 완료 (RAG 호환): {main_category} ({len(exported_data)}개 카테고리, {faqs_count}개 FAQ)")
                
            except Exception as e:
                logger.error(f"❌ {main_category} 내보내기 실패: {e}")
                continue
        
        if not exported_files:
            raise HTTPException(status_code=500, detail="데이터 내보내기에 실패했습니다.")
        
        return {
            "success": True,
            "message": f"벡터 DB의 모든 데이터를 RAG 서버 호환 JSON 파일로 내보냈습니다.",
            "exported_files": exported_files,
            "summary": {
                "total_main_categories": len(exported_files),
                "total_categories": total_categories,
                "total_faqs": total_faqs,
                "output_directory": data_json_dir
            },
            "exported_at": datetime.now(pytz.timezone('Asia/Seoul')).isoformat(),
            "rag_compatibility": "완전 호환",
            "next_steps": [
                "포트 5000의 RAG 서버에서 curl -X POST 'http://0.0.0.0:5000/api/rebuild_index'를 호출하여 데이터를 로드하세요.",
                "RAG 서버가 자동으로 ./data_json 디렉토리의 모든 JSON 파일을 인식하고 로드합니다.",
                "로드 완료 후 포트 5000에서 고급 검색 및 대화 기능을 사용할 수 있습니다."
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 전체 데이터 내보내기 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"전체 데이터 내보내기 실패: {str(e)}")

@app.get("/api/health", summary="시스템 상태 확인 (RAG 서버 호환)")
async def health_check():
    """시스템 헬스 체크 (RAG 서버 호환)"""
    try:
        # LLM API 연결 테스트
        test_response = requests.post(
            llm_client.api_url,
            json={
                "model": "qwen3-30b-a3b-mlx",
                "messages": [{"role": "user", "content": "테스트"}],
                "temperature": 0.1,
                "max_tokens": 10,
                "stream": False
            },
            timeout=10
        )
        
        llm_available = test_response.status_code == 200
        
        # 저장 디렉토리 체크
        save_directory = "./extracted_regulations"
        directory_exists = os.path.exists(save_directory)
        if not directory_exists:
            os.makedirs(save_directory, exist_ok=True)
        
        # 벡터 DB 상태 확인 (RAG 서버 호환)
        vector_db_available = False
        vector_db_stats = {}
        
        try:
            if vector_db_system is None:
                initialize_vector_db()
            if vector_db_system is not None:
                vector_db_stats = vector_db_system.get_stats()
                vector_db_available = vector_db_stats.get('vector_db_ready', False)
                logger.info(f"📁 ChromaDB 상태 확인 완료 (RAG 호환): {vector_db_stats.get('chroma_db_path', './chroma_db')}")
        except Exception as e:
            logger.warning(f"벡터 DB 상태 확인 실패: {e}")
        
        overall_status = "healthy" if (llm_available and vector_db_available) else "partial"
        
        return {
            "status": overall_status,
            "llm_api_available": llm_available,
            "llm_api_url": llm_client.api_url,
            "vector_db_available": vector_db_available,
            "vector_db_path": "./chroma_db",
            "vector_db_stats": vector_db_stats,
            "save_directory_available": True,
            "save_directory_path": save_directory,
            "rag_compatibility": "완전 호환",
            "features": [
                "PDF 텍스트 추출",
                "AI 기반 구조 분석", 
                "질문-답변 자동 생성",
                "카테고리 자동 분류",
                "JSON 구조화 출력",
                "파일 저장 및 관리",
                "벡터 DB 저장 (RAG 호환)",
                "의미 기반 검색"
            ],
            "extraction_capabilities": [
                "10가지 질문 패턴 지원",
                "다중 카테고리 자동 분류",
                "원문 기반 정확한 답변",
                "청크 단위 대용량 처리",
                "중복 제거 및 병합",
                "Form Data & JSON Body 지원",
                "강화된 에러 처리"
            ],
            "vector_db_capabilities": [
                "ChromaDB 기반 벡터 저장 (경로: ./chroma_db)",
                "SentenceTransformer 임베딩",
                "RAG 서버 호환 청킹 전략 (qa_full, question_focused, answer_focused, sentence_level)",
                "메타데이터 기반 필터링",
                "의미 기반 검색",
                "실시간 통계 조회",
                "RAG 서버 완전 호환"
            ],
            "api_endpoints": {
                "extraction": "/api/extract_regulation_json",
                "extraction_v2": "/api/extract_regulation_json_v2",
                "save_json": "/api/save_extracted_json_v2",
                "store_file_to_vector": "/api/store_json_to_vector",
                "store_direct_to_vector": "/api/store_extracted_json_to_vector",
                "vector_search": "/api/vector_search",
                "vector_stats": "/api/vector_stats",
                "export_vector_data": "/api/export_vector_to_json/{main_category}",
                "export_all_vector_data": "/api/export_all_vector_to_json",
                "file_list": "/api/saved_files",
                "file_download": "/api/download_file/{filename}",
                "file_delete": "/api/delete_file/{filename}",
                "health": "/api/health"
            },
            "timestamp": datetime.now(pytz.timezone('Asia/Seoul')).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ 헬스 체크 실패: {e}")
        return {
            "status": "unhealthy",
            "llm_api_available": False,
            "vector_db_available": False,
            "error": str(e),
            "timestamp": datetime.now(pytz.timezone('Asia/Seoul')).isoformat()
        }

# 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    logger.info("🚀 시스템 시작 중... (RAG 서버 호환)")
    # 벡터 DB는 필요할 때 초기화

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("🔥 회사 내규 PDF JSON 추출 + 벡터 DB 통합 API 서버 v3.1 (RAG 서버 호환)")
    print("=" * 80)
    print("🎯 주요 기능:")
    print("   📄 PDF 업로드 및 자동 파싱")
    print("   🧠 AI 기반 내용 구조 분석")  
    print("   📊 표준 JSON 형태로 변환")
    print("   🎯 질문-답변 자동 생성")
    print("   📝 카테고리 자동 분류")
    print("   💾 파일 저장 및 관리")
    print("   🔍 벡터 DB 저장 및 검색 (RAG 서버 완전 호환)")
    print("=" * 80)
    print("🔧 API 엔드포인트:")
    print("   - POST /api/extract_regulation_json       : 📄 PDF → JSON 변환")
    print("   - POST /api/extract_regulation_json_v2    : 📄 PDF → JSON (벡터 DB용)")
    print("   - POST /api/save_extracted_json_v2        : 💾 JSON 저장")
    print("   - POST /api/store_json_to_vector          : 🔍 파일 → 벡터 DB (RAG 호환)")
    print("   - POST /api/store_extracted_json_to_vector: 🔍 JSON → 벡터 DB (RAG 호환)")
    print("   - POST /api/vector_search                 : 🔍 벡터 검색 (RAG 호환)")
    print("   - GET  /api/vector_stats                  : 📊 벡터 DB 통계 (RAG 호환)")
    print("   - GET  /api/export_vector_to_json/{category}: 📤 벡터 DB → JSON (RAG 호환)")
    print("   - GET  /api/export_all_vector_to_json     : 📤 전체 벡터 DB → JSON (RAG 호환)")
    print("   - GET  /api/saved_files                   : 📁 파일 목록 조회")
    print("   - GET  /api/download_file/{filename}      : ⬇️ 파일 다운로드")
    print("   - DELETE /api/delete_file/{filename}      : 🗑️ 파일 삭제")
    print("   - GET  /api/health                        : ❤️ 시스템 상태 확인 (RAG 호환)")
    print("   - GET  /docs                              : 📖 API 문서")
    print("=" * 80)
    print("🔍 벡터 DB 기능 (RAG 서버 완전 호환):")
    print("   • ChromaDB 기반 고성능 벡터 저장 (저장 위치: ./chroma_db)")
    print("   • SentenceTransformer 한국어 임베딩 (nlpai-lab/KURE-v1)")
    print("   • RAG 서버와 동일한 청킹 전략:")
    print("     - qa_full: 질문+답변 전체")
    print("     - question_focused: 질문 중심")
    print("     - answer_focused: 답변 중심")
    print("     - sentence_level: 문장 단위")
    print("   • 메타데이터 기반 필터링")
    print("   • 의미 기반 검색")
    print("   • 실시간 통계 조회")
    print("   • RAG 서버 포트 5000과 완전 호환")
    print("=" * 80)
    print("🧠 AI 분석 특징:")
    print("   • 10가지 질문 패턴으로 포괄적 분석")
    print("   • 원문 기반 정확한 답변 생성")
    print("   • 의미 기반 카테고리 자동 분류")
    print("   • 대용량 문서 청크 단위 처리")
    print("   • 중복 제거 및 스마트 병합")
    print("   • Form Data & JSON Body 동시 지원")
    print("   • 강화된 에러 처리 및 로깅")
    print("=" * 80)
    print("📝 워크플로우 (RAG 서버 호환):")
    print("   1. PDF 업로드 → JSON 추출")
    print("   2. JSON 파일 저장 (선택사항)")
    print("   3. 벡터 DB에 저장 (RAG 서버와 동일한 형식)")
    print("   4. 의미 기반 검색")
    print("   5. RAG 서버로 데이터 내보내기")
    print("   ★ 신규: PDF → JSON (벡터용) → 벡터 DB (RAG 호환) (원스톱)")
    print("   ★ 호환: 벡터 DB → JSON → RAG 서버 포트 5000 (완전 호환)")
    print("=" * 80)
    print("🔧 RAG 서버 연동 방법:")
    print("   1. 이 서버(포트 5001)에서 PDF → JSON → 벡터 DB 저장")
    print("   2. /api/export_all_vector_to_json 호출하여 ./data_json에 JSON 파일 생성")
    print("   3. RAG 서버(포트 5000)에서 curl -X POST 'http://localhost:5000/api/rebuild_index' 호출")
    print("   4. RAG 서버에서 고급 검색 및 대화 기능 사용")
    print("=" * 80)
    print("🎯 청킹 전략 호환성:")
    print("   • RAG 서버와 100% 동일한 청킹 방식 사용")
    print("   • 메타데이터 구조 완전 일치")
    print("   • 임베딩 모델 동일 (nlpai-lab/KURE-v1)")
    print("   • ChromaDB 컬렉션 구조 동일")
    print("=" * 80)
    print("💡 사용 예시:")
    print("   1. http://localhost:5001/docs 접속")
    print("   2. PDF 업로드 → JSON 추출")
    print("   3. JSON을 벡터 DB에 저장 (RAG 호환)")
    print("   4. 의미 기반 검색 테스트")
    print("   5. RAG 서버로 데이터 내보내기")
    print("   ★ 원스톱: /api/extract_regulation_json_v2 → /api/store_extracted_json_to_vector")
    print("   ★ RAG 연동: /api/export_all_vector_to_json → RAG 서버 재로드")
    print("=" * 80)
    print("🔍 호환성 보장:")
    print("   • 동일한 모델: nlpai-lab/KURE-v1")
    print("   • 동일한 청킹: qa_full, question_focused, answer_focused, sentence_level")
    print("   • 동일한 메타데이터: main_category, sub_category, question, answer, source_file, chunk_type")
    print("   • 동일한 컬렉션: company_regulations")
    print("   • 동일한 저장소: ./chroma_db")
    print("=" * 80)
    print("⚠️ 중요 사항:")
    print("   • 이 서버는 RAG 서버(포트 5000)와 완전 호환됩니다")
    print("   • 벡터 DB 데이터는 양방향 호환 가능합니다")
    print("   • JSON 내보내기로 RAG 서버에 데이터 전송 가능")
    print("   • 동일한 ChromaDB 경로(./chroma_db) 사용")
    print("=" * 80)
    
    # 시스템 초기화
    logger.info("🔍 RAG 서버 호환 시스템 초기화 중...")
    
    logger.info("✅ RAG 서버 호환 FastAPI 서버 시작 중...")
    
    # FastAPI 앱 실행
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")