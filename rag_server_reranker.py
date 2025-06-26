import json
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import requests
import logging
import uuid
import time
import re
import uvicorn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from datetime import datetime  # 추가
import pytz  # 추가 (한국 시간대용, pip install pytz 필요)

# 로깅 설정 (import 직후에 설정)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Transformers 관련 임포트 (선택적 - LM Studio 사용으로 필수 아님)
try:
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
    logger.info("ℹ️ Transformers 라이브러리 사용 가능 (선택사항)")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.info("ℹ️ Transformers 라이브러리 없음 (LM Studio 사용으로 필수 아님)")

# RerankerType Enum 정의
class RerankerType(str, Enum):
    qwen3 = "qwen3"
    llm_api = "llm_api"
    sentence_transformer = "sentence_transformer"

# Pydantic 모델 정의
class SearchRequest(BaseModel):
    query: str = Field(..., description="검색할 질의", example="휴가 신청 방법")
    top_k: int = Field(20, description="반환할 결과 수", example=20, ge=1, le=50)
    main_category_filter: Optional[str] = Field(None, description="대분류 필터", example="인사")

class ConversationMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (user 또는 assistant)", example="user")
    content: str = Field(..., description="메시지 내용", example="휴가 신청은 어떻게 하나요?")
    context: Optional[List[Dict[str, Any]]] = Field(None, description="응답 컨텍스트 (assistant 메시지인 경우)")

class ChatRequest(BaseModel):
    query: str = Field(..., description="질문 내용", example="21년차 휴가는 며칠인가요?", min_length=1)
    main_category_filter: Optional[str] = Field(None, description="대분류 필터", example="인사")
    use_reranker: bool = Field(True, description="고급 검색 사용 여부 (기본: True)", example=True)
    reranker_type: RerankerType = Field(RerankerType.qwen3, description="검색 방법 선택")
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list, 
        description="이전 대화 기록"
    )

class StreamChatRequest(BaseModel):
    query: str = Field(..., description="질문 내용", example="21년차 휴가는 며칠인가요?", min_length=1)
    main_category_filter: Optional[str] = Field(None, description="대분류 필터", example="인사")
    use_reranker: bool = Field(True, description="고급 검색 사용 여부 (기본: True)", example=True)
    reranker_type: RerankerType = Field(RerankerType.qwen3, description="검색 방법 선택")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="이전 대화 기록 (유연한 형식)"
    )

class SimpleTestRequest(BaseModel):
    query: str = Field(..., description="질문", example="휴가 신청 방법", min_length=1)
    category: Optional[str] = Field(None, description="카테고리 필터", example="인사")
    use_reranker: bool = Field(True, description="고급 검색 사용 여부 (기본: True)", example=True)
    reranker_type: RerankerType = Field(RerankerType.qwen3, description="검색 방법 선택")

class SearchResult(BaseModel):
    main_category: str
    sub_category: str
    question: str
    answer: str
    source_file: str
    chunk_type: str
    score: float
    rank: int
    # rerank_details 필드 제거 - 사용자에게 노출하지 않음

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

class AdvancedReranker:
    """고급 재랭킹 시스템 - LLM API, LM Studio Qwen3-Reranker, Sentence-Transformer 지원"""
    
    def __init__(self, sentence_model: SentenceTransformer, llm_api_url: str = "http://localhost:1234/v1/chat/completions"):
        self.sentence_model = sentence_model
        self.llm_api_url = llm_api_url
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.document_texts = []
        self.qwen3_working_model = None  # 작동하는 모델명 캐시
        
        logger.info("🎯 고급 재랭킹 시스템 초기화 완료 (LM Studio Qwen3-Reranker 지원)")
        
        # 초기화 시 사용 가능한 모델 확인
        self._check_lm_studio_availability()
    
    def _check_lm_studio_availability(self) -> bool:
        """LM Studio 사용 가능 여부 및 모델 확인"""
        try:
            # 1. LM Studio 서버 연결 확인
            response = requests.get("http://localhost:1234/v1/models", timeout=5)
            if response.status_code != 200:
                logger.warning("⚠️ LM Studio 서버에 연결할 수 없습니다 (http://localhost:1234)")
                return False
            
            # 2. 사용 가능한 모델 목록 확인
            models_data = response.json()
            available_models = [model['id'] for model in models_data.get('data', [])]
            logger.info(f"📋 LM Studio 사용 가능한 모델들: {available_models}")
            
            # 3. 재랭킹 모델 찾기 및 테스트
            potential_reranker_models = [
                "qwen.qwen3-reranker-0.6b"
            ]
            
            # 4. 실제 모델명에서 재랭킹 모델 찾기
            for model in available_models:
                if 'rerank' in model.lower():
                    potential_reranker_models.insert(0, model)  # 앞에 추가
            
            # 5. 각 모델 테스트
            for model_name in potential_reranker_models:
                if model_name in available_models or model_name.startswith('qwen'):
                    if self._test_qwen3_model(model_name):
                        self.qwen3_working_model = model_name
                        logger.info(f"✅ 작동하는 Qwen3-Reranker 모델 발견: {model_name}")
                        return True
            
            logger.warning("⚠️ 작동하는 Qwen3-Reranker 모델을 찾을 수 없습니다")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ LM Studio 연결 확인 실패: {e}")
            return False
    
    def _test_qwen3_model(self, model_name: str) -> bool:
        """특정 Qwen3 모델 테스트"""
        try:
            logger.info(f"🧪 {model_name} 테스트 중...")
            
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "Answer with 'yes' if you can understand this message."}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 20,
                    "stream": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                logger.info(f"✅ {model_name} 테스트 응답: '{content}'")
                
                # 응답이 비어있지 않으면 성공
                if content and len(content) > 0:
                    return True
                else:
                    logger.warning(f"❌ {model_name}: 빈 응답")
                    return False
            else:
                logger.warning(f"❌ {model_name} 테스트 실패: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"❌ {model_name} 테스트 오류: {e}")
            return False
    
    def calculate_qwen3_similarity(self, query: str, document: str) -> float:
        """LM Studio의 Qwen3-Reranker를 사용한 의미적 유사도 계산 (개선된 버전)"""
        try:
            # 작동하는 모델이 없으면 fallback
            if not self.qwen3_working_model:
                logger.warning("⚠️ 작동하는 Qwen3 모델이 없음, fallback 사용")
                return self.calculate_semantic_similarity(query, document)
            
            # 더 간단하고 명확한 프롬프트
            system_prompt = "You are a relevance judge. Respond only with 'yes' if the document is relevant to the query, or 'no' if not relevant. Do not explain."
            
            user_prompt = f"Query: {query[:200]}\nDocument: {document[:300]}\nRelevant?"
            
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "model": self.qwen3_working_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 30,  # 충분한 토큰
                    "stream": False,
                    "stop": ["\n", "Query:", "Document:"]  # 적절한 stop 토큰
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 디버깅 로그 (과도한 로그 방지)
                if len(content) == 0:
                    logger.debug(f"🔍 빈 응답 - 모델: {self.qwen3_working_model}")
                
                if content:
                    # 태그 제거 및 정리
                    import re
                    cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    cleaned_content = re.sub(r'<[^>]*>', '', cleaned_content)
                    cleaned_content = cleaned_content.strip().lower()
                    
                    # yes/no 판단 (더 유연한 매칭)
                    if any(word in cleaned_content for word in ['yes', 'relevant', 'related', 'match', 'similar']):
                        return 0.85
                    elif any(word in cleaned_content for word in ['no', 'not', 'irrelevant', 'unrelated', 'different']):
                        return 0.15
                    else:
                        # 응답이 있지만 yes/no가 명확하지 않은 경우
                        logger.debug(f"🟡 애매한 응답: '{cleaned_content[:50]}'")
                        return 0.5
                else:
                    # 빈 응답 처리 - 과도한 로그 방지
                    logger.debug(f"⚠️ 빈 응답 from {self.qwen3_working_model}")
                    return 0.5
            else:
                logger.warning(f"⚠️ Qwen3-Reranker API 오류: {response.status_code}")
                return self.calculate_semantic_similarity(query, document)
                
        except requests.exceptions.Timeout:
            logger.warning("⚠️ Qwen3-Reranker 타임아웃")
            return self.calculate_semantic_similarity(query, document)
        except Exception as e:
            logger.warning(f"⚠️ Qwen3-Reranker 오류: {e}")
            return self.calculate_semantic_similarity(query, document)
    
    def calculate_llm_api_similarity(self, query: str, document: str) -> float:
        """LLM API를 사용한 의미적 유사도 계산"""
        try:
            # 매우 명확하고 강력한 프롬프트
            system_prompt = """You are a precise scoring system. Your ONLY job is to output a single decimal number between 0.0 and 1.0.

CRITICAL RULES:
- DO NOT use <think> tags
- DO NOT explain your reasoning  
- DO NOT write any text
- OUTPUT ONLY a number like: 0.7

Rate document relevance to query:
0.0-0.2: Irrelevant
0.3-0.4: Somewhat related
0.5-0.6: Partially relevant  
0.7-0.8: Highly relevant
0.9-1.0: Perfect match

Examples of CORRECT responses: 0.8, 0.3, 0.9"""

            user_prompt = f"""Query: {query}
Document: {document}
Score:"""

            response = requests.post(
                self.llm_api_url,
                json={
                    "model": "qwen3-30b-a3b-mlx",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0,  # 완전히 결정적으로
                    "max_tokens": 5,     # 매우 짧게
                    "stream": False,
                    "stop": ["\n", "<", " ", ".", ",", "think"]  # <think> 방지
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 1. <think> 태그 및 기타 태그 제거
                import re
                cleaned_content = re.sub(r'<[^>]*>', '', content)  # 모든 HTML/XML 태그 제거
                cleaned_content = cleaned_content.strip()
                
                # 2. 숫자 패턴 추출 (더 강력하게)
                number_patterns = [
                    r'\b([01]\.?\d*)\b',      # 0.xxx 또는 1.xxx
                    r'\b(0\.\d+)\b',          # 0.xxx
                    r'\b(1\.0*)\b',           # 1 또는 1.0
                    r'\b(0)\b'                # 0
                ]
                
                for pattern in number_patterns:
                    matches = re.findall(pattern, cleaned_content)
                    if matches:
                        try:
                            score = float(matches[0])
                            if 0.0 <= score <= 1.0:
                                return score
                            elif 1.0 < score <= 10.0:  # 1-10 스케일 변환
                                return score / 10.0
                        except ValueError:
                            continue
                
                # 3. 원본 content에서 직접 숫자 찾기
                all_numbers = re.findall(r'\d*\.?\d+', content)
                for num_str in all_numbers:
                    try:
                        score = float(num_str)
                        if 0.0 <= score <= 1.0:
                            return score
                        elif 1.0 < score <= 10.0:
                            return score / 10.0
                    except ValueError:
                        continue
                
                # 4. 길이 기반 추정 (응급 처치)
                if len(cleaned_content) <= 3:  # 매우 짧으면 숫자일 가능성
                    try:
                        score = float(cleaned_content)
                        if 0.0 <= score <= 1.0:
                            return score
                    except ValueError:
                        pass
                
                # 5. 키워드 기반 점수 (최후 수단)
                content_lower = content.lower()
                if any(word in content_lower for word in ['perfect', 'excellent', '1.0', '0.9']):
                    return 0.9
                elif any(word in content_lower for word in ['good', 'relevant', '0.8', '0.7']):
                    return 0.7
                elif any(word in content_lower for word in ['partial', 'some', '0.5', '0.6']):
                    return 0.5
                elif any(word in content_lower for word in ['little', 'weak', '0.3', '0.4']):
                    return 0.3
                elif any(word in content_lower for word in ['irrelevant', 'unrelated', '0.0', '0.1']):
                    return 0.1
                
                # 로그에 더 많은 정보 출력
                logger.warning(f"⚠️ LLM API 점수 파싱 실패 - 원본: '{content[:50]}', 정리: '{cleaned_content[:30]}'")
                return 0.5  # 기본값
            else:
                logger.warning(f"⚠️ LLM API 호출 실패: {response.status_code}")
                return 0.5
                
        except requests.exceptions.Timeout:
            logger.warning("⚠️ LLM API 타임아웃")
            return 0.5
        except Exception as e:
            logger.warning(f"⚠️ LLM API 유사도 계산 실패: {e}")
            return 0.5
    
    def calculate_semantic_similarity(self, query: str, document: str) -> float:
        """Sentence-Transformer를 사용한 의미적 유사도 계산 (기본 방법)"""
        try:
            query_embedding = self.sentence_model.encode([query])
            doc_embedding = self.sentence_model.encode([document])
            
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            return max(0.0, float(similarity))
        except Exception as e:
            logger.warning(f"⚠️ 의미적 유사도 계산 실패: {e}")
            return 0.0
    
    def get_semantic_similarity(self, query: str, document: str, reranker_type: str = "sentence_transformer") -> float:
        """재랭킹 방법에 따른 의미적 유사도 계산"""
        if reranker_type == "llm_api":
            return self.calculate_llm_api_similarity(query, document)
        elif reranker_type == "qwen3":
            return self.calculate_qwen3_similarity(query, document)
        else:  # sentence_transformer (기본)
            return self.calculate_semantic_similarity(query, document)
    
    def prepare_tfidf_index(self, documents: List[str]):
        """TF-IDF 인덱스 준비 (키워드 매칭용)"""
        try:
            self.document_texts = documents
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=None,  # 한국어는 별도 처리
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            logger.info(f"✅ TF-IDF 인덱스 구축 완료: {len(documents)}개 문서, {self.tfidf_matrix.shape[1]}개 특성")
        except Exception as e:
            logger.error(f"❌ TF-IDF 인덱스 구축 실패: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def calculate_keyword_relevance(self, query: str, document: str) -> float:
        """키워드 관련성 계산 (TF-IDF 기반)"""
        try:
            if self.tfidf_vectorizer is None:
                return 0.0
            
            query_vector = self.tfidf_vectorizer.transform([query])
            doc_vector = self.tfidf_vectorizer.transform([document])
            
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            return max(0.0, float(similarity))
        except Exception as e:
            logger.warning(f"⚠️ 키워드 관련성 계산 실패: {e}")
            return 0.0
    
    def calculate_structural_relevance(self, query: str, candidate: Dict[str, Any]) -> float:
        """구조적 관련성 계산 (질문-답변 구조 고려)"""
        try:
            question = candidate.get('question', '').lower()
            answer = candidate.get('answer', '').lower()
            query_lower = query.lower()
            
            # 질문 매칭 점수
            question_match = self._calculate_text_overlap(query_lower, question)
            
            # 답변 매칭 점수
            answer_match = self._calculate_text_overlap(query_lower, answer)
            
            # 청크 타입에 따른 가중치
            chunk_type = candidate.get('chunk_type', 'qa_full')
            chunk_weights = {
                'qa_full': 1.0,
                'question_focused': 0.95,
                'answer_focused': 0.85,
                'sentence_level': 0.75
            }
            chunk_weight = chunk_weights.get(chunk_type, 0.5)
            
            # 종합 점수 계산
            structural_score = (question_match * 0.6 + answer_match * 0.4) * chunk_weight
            return min(1.0, structural_score)
            
        except Exception as e:
            logger.warning(f"⚠️ 구조적 관련성 계산 실패: {e}")
            return 0.0
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """텍스트 간 중복도 계산"""
        try:
            words1 = set(re.findall(r'\b[가-힣]{2,}\b', text1))
            words2 = set(re.findall(r'\b[가-힣]{2,}\b', text2))
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            coverage = intersection / len(words1) if len(words1) > 0 else 0.0
            
            return (jaccard_similarity + coverage) / 2
        except Exception as e:
            return 0.0
    
    def calculate_context_relevance(self, query: str, candidate: Dict[str, Any], 
                                  conversation_history: List[Dict[str, str]] = None) -> float:
        """컨텍스트 관련성 계산 (이전 대화 고려)"""
        try:
            if not conversation_history:
                return 0.0
            
            # 이전 대화에서 언급된 키워드 추출
            context_keywords = set()
            for msg in conversation_history[-3:]:  # 최근 3개 메시지만
                content = msg.get('content', '').lower()
                keywords = re.findall(r'\b[가-힣]{2,}\b', content)
                context_keywords.update(keywords)
            
            if not context_keywords:
                return 0.0
            
            # 후보 문서에서 컨텍스트 키워드 매칭
            question = candidate.get('question', '').lower()
            answer = candidate.get('answer', '').lower()
            doc_text = f"{question} {answer}"
            doc_keywords = set(re.findall(r'\b[가-힣]{2,}\b', doc_text))
            
            # 컨텍스트 매칭 점수
            matches = len(context_keywords.intersection(doc_keywords))
            context_score = matches / len(context_keywords) if len(context_keywords) > 0 else 0.0
            
            return min(1.0, context_score * 2)  # 부스트 적용
            
        except Exception as e:
            logger.warning(f"⚠️ 컨텍스트 관련성 계산 실패: {e}")
            return 0.0
    
    def calculate_answer_quality_score(self, candidate: Dict[str, Any]) -> float:
        """답변 품질 점수 계산"""
        try:
            answer = candidate.get('answer', '')
            
            # 답변 길이 점수 (너무 짧거나 너무 길면 감점)
            length = len(answer)
            if length < 10:
                length_score = 0.3
            elif length < 50:
                length_score = 0.6
            elif length < 200:
                length_score = 1.0
            elif length < 500:
                length_score = 0.9
            else:
                length_score = 0.7
            
            # 구조적 요소 점수 (숫자, 단계, 예시 등)
            structure_score = 0.5
            if re.search(r'\d+\.\s', answer):  # 번호 목록
                structure_score += 0.2
            if '단계' in answer or '절차' in answer:  # 절차 설명
                structure_score += 0.2
            if '예:' in answer or '예시' in answer:  # 예시 포함
                structure_score += 0.1
            
            structure_score = min(1.0, structure_score)
            
            # 종합 점수
            quality_score = (length_score * 0.7) + (structure_score * 0.3)
            return quality_score
            
        except Exception as e:
            logger.warning(f"⚠️ 답변 품질 점수 계산 실패: {e}")
            return 0.5
    
    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], 
                         conversation_history: List[Dict[str, str]] = None,
                         top_k: int = 20, reranker_type: str = "sentence_transformer") -> List[Dict[str, Any]]:
        """고급 재랭킹 수행 - 다중 방법 지원"""
        logger.info(f"🎯 고급 재랭킹 시작 ({reranker_type}): {len(candidates)}개 후보 → {top_k}개 선별")
        
        # Qwen3 상태 확인
        if reranker_type == "qwen3" and not self.qwen3_working_model:
            logger.warning("⚠️ Qwen3 모델 사용 불가, sentence_transformer로 변경")
            reranker_type = "sentence_transformer"
        
        try:
            if not candidates:
                return []
            
            # 문서 텍스트 준비 (TF-IDF용)
            documents = [f"{c.get('question', '')} {c.get('answer', '')}" for c in candidates]
            self.prepare_tfidf_index(documents)
            
            reranked_candidates = []
            
            for i, candidate in enumerate(candidates):
                # 1. 의미적 유사도 (선택된 방법 사용)
                doc_text = f"{candidate.get('question', '')} {candidate.get('answer', '')}"
                semantic_score = self.get_semantic_similarity(query, doc_text, reranker_type)
                
                # 2. 키워드 관련성 (TF-IDF 기반)
                keyword_score = self.calculate_keyword_relevance(query, doc_text)
                
                # 3. 구조적 관련성 (질문-답변 구조)
                structural_score = self.calculate_structural_relevance(query, candidate)
                
                # 4. 컨텍스트 관련성 (이전 대화 고려)
                context_score = self.calculate_context_relevance(query, candidate, conversation_history)
                
                # 5. 답변 품질 점수
                quality_score = self.calculate_answer_quality_score(candidate)
                
                # 6. 원본 벡터 검색 점수
                original_score = candidate.get('score', 0.0)
                
                # 다중 점수 융합 (가중 평균)
                final_score = (
                    semantic_score * 0.30 +      # 의미적 유사도 (가장 중요)
                    structural_score * 0.25 +    # 구조적 관련성
                    keyword_score * 0.20 +       # 키워드 매칭
                    original_score * 0.15 +      # 원본 벡터 점수
                    quality_score * 0.05 +       # 답변 품질
                    context_score * 0.05         # 컨텍스트 관련성
                )
                
                # 재랭킹 상세 정보 저장 (내부용 - 사용자에게는 노출하지 않음)
                rerank_details = {
                    'semantic_score': round(semantic_score, 3),
                    'keyword_score': round(keyword_score, 3),
                    'structural_score': round(structural_score, 3),
                    'context_score': round(context_score, 3),
                    'quality_score': round(quality_score, 3),
                    'original_score': round(original_score, 3),
                    'final_score': round(final_score, 3),
                    'reranker_type': reranker_type
                }
                
                # 후보 업데이트
                enhanced_candidate = candidate.copy()
                enhanced_candidate['score'] = final_score
                enhanced_candidate['rerank_details'] = rerank_details  # 내부적으로만 사용
                
                reranked_candidates.append(enhanced_candidate)
            
            # 최종 점수로 정렬
            reranked_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 다양성 확보 (같은 질문의 중복 제거)
            diverse_results = []
            seen_questions = set()
            
            for candidate in reranked_candidates:
                question = candidate.get('question', '')
                if question not in seen_questions or len(diverse_results) < top_k // 2:
                    diverse_results.append(candidate)
                    seen_questions.add(question)
                
                if len(diverse_results) >= top_k:
                    break
            
            # 순위 재할당
            for i, result in enumerate(diverse_results):
                result['rank'] = i + 1
            
            logger.info(f"✅ 고급 재랭킹 완료 ({reranker_type}): {len(diverse_results)}개 최종 선별")
            
            # 재랭킹 성능 로그
            if diverse_results:
                avg_semantic = np.mean([r['rerank_details']['semantic_score'] for r in diverse_results])
                avg_structural = np.mean([r['rerank_details']['structural_score'] for r in diverse_results])
                avg_final = np.mean([r['score'] for r in diverse_results])
                logger.info(f"📊 재랭킹 품질 ({reranker_type}): 의미적={avg_semantic:.3f}, 구조적={avg_structural:.3f}, 최종={avg_final:.3f}")
            
            return diverse_results
            
        except Exception as e:
            logger.error(f"❌ 재랭킹 실패: {e}", exc_info=True)
            # 실패 시 원본 결과 반환
            return candidates[:top_k]
    
    def get_status(self) -> Dict[str, Any]:
        """재랭킹 시스템 상태 반환"""
        return {
            "qwen3_available": self.qwen3_working_model is not None,
            "qwen3_model": self.qwen3_working_model,
            "sentence_transformer_available": self.sentence_model is not None,
            "tfidf_ready": self.tfidf_vectorizer is not None
        }

class CompanyRegulationsRAGSystem:
    def __init__(self, model_name: str = "nlpai-lab/KURE-v1", persist_directory: str = "./chroma_db"):
        """고급 검색 기반 ChromaDB RAG 시스템 초기화"""
        model_path = os.path.join("./models", model_name.replace("/", "-"))

        # 모델 로드
        if not os.path.exists(model_path):
            logger.info(f"📦 모델 다운로드: '{model_name}'")
            try:
                model = SentenceTransformer(model_name)
                model.save(model_path)
                logger.info(f"✅ 모델 저장 완료: {model_path}")
            except Exception as e:
                logger.error(f"❌ 모델 다운로드 실패: {e}", exc_info=True)
                raise
        else:
            logger.info(f"🔄 기존 모델 로드: {model_path}")

        self.model = SentenceTransformer(model_path)
        self.persist_directory = persist_directory
        
        # 고급 재랭킹 시스템 초기화
        self.reranker = AdvancedReranker(self.model)

        # ChromaDB 초기화
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            self.collection = self.chroma_client.get_collection(name="company_regulations")
            logger.info("📁 기존 company_regulations 컬렉션 로드 완료")
        except Exception as get_error:
            logger.info(f"📁 새 컬렉션 생성 중...")
            try:
                self.collection = self.chroma_client.create_collection(
                    name="company_regulations",
                    metadata={"description": "고급 검색 기반 회사 내규 벡터 검색 컬렉션"}
                )
                logger.info("📁 새 컬렉션 생성 완료")
            except Exception as create_error:
                import time
                temp_name = f"company_regulations_advanced_{int(time.time())}"
                self.collection = self.chroma_client.create_collection(
                    name=temp_name,
                    metadata={"description": "고급 검색 기반 임시 컬렉션"}
                )
                logger.info(f"📁 임시 컬렉션 '{temp_name}' 생성 완료")

        logger.info(f"✅ 고급 검색 기반 RAG 시스템 초기화 완료")
    
    def load_company_regulations_data(self, data_directory: str = "./data_json"):
        """회사 내규 데이터 로드"""
        logger.info(f"회사 내규 데이터 로드 시작: {data_directory}")
        try:
            if not os.path.exists(data_directory):
                logger.error(f"❌ 데이터 디렉토리 없음: {data_directory}")
                return False
            
            json_files = glob.glob(os.path.join(data_directory, "*.json"))
            
            if not json_files:
                logger.warning(f"⚠️ JSON 파일 없음: {data_directory}")
                return False
            
            logger.info(f"🔍 발견된 JSON 파일: {len(json_files)}개")
            
            self.regulations_data = []
            self.main_categories = {}
            
            for json_file in json_files:
                file_name = os.path.basename(json_file)
                main_category = os.path.splitext(file_name)[0]
                
                logger.info(f"📂 로딩: {file_name} → {main_category}")
                
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
                            
                            # 향상된 청킹
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
                            
                            # 긴 답변의 경우 문장 분할
                            if len(answer) > 200:
                                sentences = re.split(r'[.!?]\s+', answer)
                                for i, sentence in enumerate(sentences):
                                    if len(sentence.strip()) > 20:
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
                    
                    logger.info(f"  → 소분류: {category_count}개, 규정: {faq_count}개")
                    
                except Exception as e:
                    logger.error(f"❌ 파일 로드 실패 {json_file}: {e}")
                    continue
            
            total_chunks = len(self.regulations_data)
            total_files = len(self.main_categories)
            
            logger.info(f"✅ 데이터 로드 완료: 파일 {total_files}개, 청크 {total_chunks}개")
            return total_chunks > 0
            
        except Exception as e:
            logger.error(f"❌ 데이터 로드 실패: {e}", exc_info=True)
            return False

    def build_index(self, force_rebuild=False):
        """ChromaDB 인덱스 구축 (force_rebuild=True일 때만 재구축)"""
        logger.info("ChromaDB 인덱스 구축 시작")
        try:
            existing_count = self.collection.count()
            
            # 기존 데이터가 있고 강제 재구축이 아닌 경우 그대로 사용
            if existing_count > 0 and not force_rebuild:
                logger.info(f"✅ 기존 벡터 DB 사용: {existing_count}개 (재구축 하지 않음)")
                return True
            
            # 강제 재구축이거나 데이터가 없는 경우에만 진행
            if not hasattr(self, 'regulations_data') or not self.regulations_data:
                if existing_count > 0:
                    logger.info(f"✅ 기존 벡터 DB 사용: {existing_count}개 (data_json 없음)")
                    return True
                else:
                    logger.warning("⚠️ 로드된 데이터와 기존 벡터 DB 모두 없음")
                    return False

            if force_rebuild and existing_count > 0:
                logger.info("🔄 강제 인덱스 재구축 중...")
                self._clear_collection_safely()
            
            texts = [item['text'] for item in self.regulations_data]
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
                for item in self.regulations_data
            ]
            ids = [item['id'] for item in self.regulations_data]
            
            logger.info(f"⚙️ 임베딩 및 저장: {len(texts)}개 청크")
            
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
            logger.info(f"✅ 인덱스 구축 완료: {final_count}개")
            return True
            
        except Exception as e:
            logger.error(f"❌ 인덱스 구축 실패: {e}", exc_info=True)
            return False
    
    def _clear_collection_safely(self):
        """컬렉션 안전 삭제"""
        try:
            # 개별 ID 삭제 시도
            try:
                all_results = self.collection.get()
                if all_results and 'ids' in all_results and all_results['ids']:
                    batch_size = 1000
                    total_ids = all_results['ids']
                    
                    for i in range(0, len(total_ids), batch_size):
                        batch_ids = total_ids[i:i+batch_size]
                        self.collection.delete(ids=batch_ids)
                return True
            except Exception:
                # 컬렉션 재생성
                collection_name = self.collection.name
                self.chroma_client.delete_collection(name=collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "고급 검색 기반 회사 내규 벡터 검색 컬렉션"}
                )
                return True
        except Exception as e:
            logger.error(f"❌ 컬렉션 정리 실패: {e}")
            return False
    
    def _generate_query_variants(self, query: str) -> List[str]:
        """쿼리 변형 생성"""
        variants = [query]  # 원본 쿼리
        
        # 의문사 제거
        question_words = ['무엇', '어떻게', '언제', '어디서', '왜', '누가', '얼마나', '몇']
        cleaned_query = query
        for word in question_words:
            cleaned_query = cleaned_query.replace(word, '').strip()
        if cleaned_query and cleaned_query != query:
            variants.append(cleaned_query)
        
        # 핵심 키워드 추출
        keywords = re.findall(r'\b[가-힣]{2,}\b', query)
        if len(keywords) >= 2:
            keyword_query = ' '.join(keywords[:3])
            variants.append(keyword_query)
        
        # 문제/절차 변형
        if '문제' in query or '이슈' in query:
            variants.append(query.replace('문제', '해결').replace('이슈', '대응'))
        
        if '방법' in query or '절차' in query:
            procedure_query = query.replace('방법', '절차 단계').replace('과정', '절차')
            variants.append(procedure_query)
        
        return list(set(variants))
    
    def search_with_advanced_rerank(self, query: str, top_k: int = 20, 
                                   main_category_filter: str = None,
                                   conversation_history: List[Dict[str, str]] = None,
                                   min_relevance_score: float = 0.2,
                                   reranker_type: str = "sentence_transformer") -> List[Dict[str, Any]]:
        """고급 재랭킹 기반 검색 - 2단계 검색 아키텍처 (다중 재랭킹 방법 지원)"""
        logger.info(f"🎯 고급 검색 실행 ({reranker_type}): '{query[:50]}...', top_k={top_k}")
        try:
            if self.collection.count() == 0:
                logger.warning("⚠️ 빈 인덱스")
                return []
            
            # 1단계: 광범위한 후보 수집 (Retrieval)
            query_variants = self._generate_query_variants(query)
            logger.info(f"🔀 쿼리 변형: {len(query_variants)}개")
            
            all_candidates = {}
            
            # 필터 조건
            where_condition = None
            if main_category_filter:
                where_condition = {"main_category": main_category_filter}
            
            # 더 많은 후보 수집 (재랭킹을 위해)
            retrieval_limit = max(top_k * 4, 80)  # 4배 더 많은 후보 수집
            
            for i, variant_query in enumerate(query_variants):
                logger.info(f"  🔍 검색 {i+1}: '{variant_query[:30]}...'")
                
                query_embedding = self.model.encode([variant_query]).tolist()
                
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=retrieval_limit,
                    include=['documents', 'metadatas', 'distances'],
                    where=where_condition
                )
                
                if results['ids'] and len(results['ids'][0]) > 0:
                    for j in range(len(results['ids'][0])):
                        doc_id = results['ids'][0][j]
                        metadata = results['metadatas'][0][j]
                        distance = results['distances'][0][j]
                        
                        similarity_score = max(0, 1 - (distance / 2))
                        query_weight = 1.0 if i == 0 else 0.8
                        weighted_score = similarity_score * query_weight
                        
                        if doc_id not in all_candidates:
                            all_candidates[doc_id] = {
                                'metadata': metadata,
                                'score': weighted_score,
                                'distances': [distance],
                                'query_matches': [i]
                            }
                        else:
                            if weighted_score > all_candidates[doc_id]['score']:
                                all_candidates[doc_id]['score'] = weighted_score
                            all_candidates[doc_id]['distances'].append(distance)
                            all_candidates[doc_id]['query_matches'].append(i)
            
            logger.info(f"  📊 수집된 후보: {len(all_candidates)}개")
            
            # 후보를 리스트로 변환
            candidate_list = []
            for doc_id, candidate_data in all_candidates.items():
                metadata = candidate_data['metadata']
                candidate = {
                    'main_category': metadata['main_category'],
                    'sub_category': metadata['sub_category'],
                    'question': metadata['question'],
                    'answer': metadata['answer'],
                    'source_file': metadata['source_file'],
                    'chunk_type': metadata.get('chunk_type', 'qa_full'),
                    'score': candidate_data['score'],
                    'query_matches': len(set(candidate_data['query_matches']))
                }
                
                # 최소 관련성 필터링
                if candidate['score'] >= min_relevance_score:
                    candidate_list.append(candidate)
            
            logger.info(f"  📊 필터링 후 후보: {len(candidate_list)}개")
            
            # 2단계: 고급 재랭킹 (Rerank)
            reranked_results = self.reranker.rerank_candidates(
                query=query,
                candidates=candidate_list,
                conversation_history=conversation_history,
                top_k=top_k,
                reranker_type=reranker_type
            )
            
            logger.info(f"✅ 고급 검색 완료 ({reranker_type}): {len(reranked_results)}개 최종 결과")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"❌ 고급 검색 실패: {e}", exc_info=True)
            return []
    
    def search_with_enhanced_retrieval(self, query: str, top_k: int = 20, 
                                      main_category_filter: str = None, 
                                      min_relevance_score: float = 0.3) -> List[Dict[str, Any]]:
        """기존 향상된 검색 (재랭킹 없이) - 다중 쿼리, 하이브리드 검색"""
        logger.info(f"향상된 검색 실행: '{query[:50]}...', top_k={top_k}")
        try:
            if self.collection.count() == 0:
                logger.warning("⚠️ 빈 인덱스")
                return []
            
            # 다중 쿼리 생성
            query_variants = self._generate_query_variants(query)
            logger.info(f"🔀 쿼리 변형: {len(query_variants)}개")
            
            all_candidates = {}
            
            # 필터 조건
            where_condition = None
            if main_category_filter:
                where_condition = {"main_category": main_category_filter}
            
            search_limit = max(top_k * 3, 60)
            
            for i, variant_query in enumerate(query_variants):
                query_embedding = self.model.encode([variant_query]).tolist()
                
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=search_limit,
                    include=['documents', 'metadatas', 'distances'],
                    where=where_condition
                )
                
                if results['ids'] and len(results['ids'][0]) > 0:
                    for j in range(len(results['ids'][0])):
                        doc_id = results['ids'][0][j]
                        metadata = results['metadatas'][0][j]
                        distance = results['distances'][0][j]
                        
                        similarity_score = max(0, 1 - (distance / 2))
                        query_weight = 1.0 if i == 0 else 0.8
                        weighted_score = similarity_score * query_weight
                        
                        if doc_id not in all_candidates:
                            all_candidates[doc_id] = {
                                'metadata': metadata,
                                'best_similarity': weighted_score,
                                'query_matches': [i],
                                'distances': [distance]
                            }
                        else:
                            if weighted_score > all_candidates[doc_id]['best_similarity']:
                                all_candidates[doc_id]['best_similarity'] = weighted_score
                            all_candidates[doc_id]['query_matches'].append(i)
                            all_candidates[doc_id]['distances'].append(distance)
            
            # 기존 재랭킹 (키워드 매칭 기반)
            enhanced_candidates = []
            
            for doc_id, candidate_data in all_candidates.items():
                metadata = candidate_data['metadata']
                best_similarity = candidate_data['best_similarity']
                query_matches = candidate_data['query_matches']
                
                question = metadata['question'].lower()
                answer = metadata['answer'].lower()
                query_lower = query.lower()
                chunk_type = metadata.get('chunk_type', 'qa_full')
                
                # 키워드 매칭 분석
                query_words = set(re.findall(r'\b[가-힣]{2,}\b', query_lower))
                question_words = set(re.findall(r'\b[가-힣]{2,}\b', question))
                answer_words = set(re.findall(r'\b[가-힣]{2,}\b', answer))
                
                question_match_count = len(query_words.intersection(question_words))
                question_match_ratio = question_match_count / max(len(query_words), 1)
                
                answer_match_count = len(query_words.intersection(answer_words))
                answer_match_ratio = answer_match_count / max(len(query_words), 1)
                
                # 청크 타입별 가중치
                chunk_weights = {
                    'qa_full': 1.0,
                    'question_focused': 0.9,
                    'answer_focused': 0.8,
                    'sentence_level': 0.7
                }
                chunk_weight = chunk_weights.get(chunk_type, 0.5)
                
                multi_query_bonus = len(set(query_matches)) * 0.1
                keyword_bonus = (question_match_ratio * 0.4) + (answer_match_ratio * 0.3)
                
                final_score = min(1.0, (best_similarity * chunk_weight) + keyword_bonus + multi_query_bonus)
                
                if final_score >= min_relevance_score:
                    enhanced_candidates.append({
                        'main_category': metadata['main_category'],
                        'sub_category': metadata['sub_category'],
                        'question': metadata['question'],
                        'answer': metadata['answer'],
                        'source_file': metadata['source_file'],
                        'chunk_type': chunk_type,
                        'score': final_score,
                        'similarity_score': best_similarity,
                        'keyword_bonus': keyword_bonus,
                        'multi_query_bonus': multi_query_bonus,
                        'chunk_weight': chunk_weight,
                        'query_matches': len(set(query_matches))
                    })
            
            # 점수 기반 정렬 및 다양성 확보
            enhanced_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 다양성을 위한 중복 제거
            seen_questions = set()
            diverse_results = []
            
            for candidate in enhanced_candidates:
                question = candidate['question']
                
                if question not in seen_questions:
                    diverse_results.append(candidate)
                    seen_questions.add(question)
                elif candidate['chunk_type'] == 'qa_full' and len(diverse_results) < top_k:
                    for i, existing in enumerate(diverse_results):
                        if existing['question'] == question and existing['chunk_type'] != 'qa_full':
                            diverse_results[i] = candidate
                            break
                
                if len(diverse_results) >= top_k:
                    break
            
            final_results = diverse_results[:top_k]
            
            # 랭크 할당
            for i, result in enumerate(final_results):
                result['rank'] = i + 1
            
            logger.info(f"✅ 향상된 검색 완료: {len(final_results)}개 결과")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 향상된 검색 실패: {e}", exc_info=True)
            return []
    
    def search(self, query: str, top_k: int = 20, main_category_filter: str = None, 
              conversation_history: List[Dict[str, str]] = None, use_reranker: bool = True,
              reranker_type: str = "sentence_transformer") -> List[Dict[str, Any]]:
        """통합 검색 메서드 - 고급 검색 사용 여부 및 방법 선택"""
        if use_reranker:
            logger.info(f"🎯 고급 검색 사용 ({reranker_type})")
            return self.search_with_advanced_rerank(
                query=query,
                top_k=top_k,
                main_category_filter=main_category_filter,
                conversation_history=conversation_history,
                reranker_type=reranker_type
            )
        else:
            logger.info(f"📊 향상된 검색 사용")
            return self.search_with_enhanced_retrieval(
                query=query,
                top_k=top_k,
                main_category_filter=main_category_filter
            )
    
    def combine_contexts_with_history(self, current_context: List[Dict[str, Any]], 
                                    conversation_history: List[Dict[str, str]], 
                                    max_total_context: int = 25) -> List[Dict[str, Any]]:
        """컨텍스트 결합"""
        logger.info(f"컨텍스트 결합: 현재 {len(current_context)}개, 최대 {max_total_context}개")
        try:
            # 이전 대화 컨텍스트 추출
            previous_contexts = []
            if conversation_history:
                for msg in conversation_history:
                    if msg.get('role') == 'assistant' and msg.get('context'):
                        previous_contexts.extend(msg['context'])
            
            # 중복 제거
            seen_questions = set()
            unique_contexts = []
            
            # 현재 검색 결과 우선 추가
            for ctx in current_context:
                question = ctx.get('question', '')
                if question and question not in seen_questions:
                    seen_questions.add(question)
                    ctx_copy = ctx.copy()
                    ctx_copy['source_type'] = 'current'
                    unique_contexts.append(ctx_copy)
            
            # 이전 컨텍스트 추가
            remaining_slots = max_total_context - len(unique_contexts)
            if remaining_slots > 0:
                previous_sorted = sorted(
                    previous_contexts, 
                    key=lambda x: x.get('score', 0), 
                    reverse=True
                )
                
                for ctx in previous_sorted:
                    if len(unique_contexts) >= max_total_context:
                        break
                    
                    question = ctx.get('question', '')
                    if question and question not in seen_questions and ctx.get('score', 0) >= 0.2:
                        seen_questions.add(question)
                        ctx_copy = ctx.copy()
                        ctx_copy['score'] = ctx_copy.get('score', 0) * 0.85
                        ctx_copy['source_type'] = 'previous'
                        unique_contexts.append(ctx_copy)
            
            # 최종 정렬
            unique_contexts.sort(key=lambda x: (x.get('source_type') == 'current', x.get('score', 0)), reverse=True)
            
            logger.info(f"✅ 컨텍스트 결합 완료: {len(unique_contexts)}개")
            
            return unique_contexts
            
        except Exception as e:
            logger.error(f"❌ 컨텍스트 결합 실패: {e}")
            return current_context

    def _natural_sort_key(self, text: str):
        """자연스러운 숫자 정렬을 위한 키 함수"""
        import re
        
        def convert(text_part):
            if text_part.isdigit():
                return int(text_part)
            else:
                return text_part.lower()
        
        # 숫자와 문자를 분리해서 정렬 키 생성
        return [convert(c) for c in re.split(r'(\d+)', text)]

    def get_categories_from_vector_db(self) -> Dict[str, Any]:
        """ChromaDB에서 직접 카테고리 정보 조회 (동적)"""
        try:
            if self.collection.count() == 0:
                return {'main_categories': {}, 'total_main_categories': 0, 'total_regulations': 0}
            
            # ChromaDB에서 모든 메타데이터 가져오기
            all_results = self.collection.get(include=['metadatas'])
            
            if not all_results or not all_results.get('metadatas'):
                return {'main_categories': {}, 'total_main_categories': 0, 'total_regulations': 0}
            
            # 대분류별 통계 계산
            main_categories = {}
            total_documents = 0
            
            for metadata in all_results['metadatas']:
                # qa_full 청크만 카운트 (중복 방지)
                if metadata.get('chunk_type') != 'qa_full':
                    continue
                    
                main_cat = metadata.get('main_category', 'Unknown')
                sub_cat = metadata.get('sub_category', 'Unknown')
                source_file = metadata.get('source_file', f'{main_cat}.json')
                
                if main_cat not in main_categories:
                    main_categories[main_cat] = {
                        'file_name': source_file,
                        'sub_categories': set(),
                        'total_faqs': 0
                    }
                
                main_categories[main_cat]['sub_categories'].add(sub_cat)
                main_categories[main_cat]['total_faqs'] += 1
                total_documents += 1
            
            # set을 개수로 변환
            for main_cat in main_categories:
                main_categories[main_cat]['sub_categories'] = len(main_categories[main_cat]['sub_categories'])
            
            # 자연스러운 숫자 정렬
            sorted_items = sorted(main_categories.items(), key=lambda x: self._natural_sort_key(x[0]))
            sorted_main_categories = dict(sorted_items)
            
            return {
                'main_categories': sorted_main_categories,
                'total_main_categories': len(main_categories),
                'total_regulations': total_documents
            }
            
        except Exception as e:
            logger.error(f"❌ ChromaDB 카테고리 조회 실패: {e}")
            # 실패 시 기존 방식으로 fallback
            return self.get_categories_from_files()

    def get_categories_from_files(self) -> Dict[str, Any]:
        """기존 파일 기반 카테고리 조회 (fallback)"""
        if not hasattr(self, 'main_categories'):
            return {'main_categories': {}, 'total_main_categories': 0, 'total_regulations': 0}
        
        # 자연스러운 숫자 정렬
        sorted_items = sorted(self.main_categories.items(), key=lambda x: self._natural_sort_key(x[0]))
        sorted_main_categories = dict(sorted_items)
        
        return {
            'main_categories': sorted_main_categories,
            'total_main_categories': len(self.main_categories),
            'total_regulations': len(self.regulations_data) if hasattr(self, 'regulations_data') else 0
        }

    def get_categories(self) -> Dict[str, Any]:
        """통합 카테고리 정보 반환 (동적 + fallback)"""
        # 1. ChromaDB에서 동적으로 조회 시도
        dynamic_categories = self.get_categories_from_vector_db()
        
        # 2. 파일 기반 카테고리와 병합
        file_categories = self.get_categories_from_files()
        
        # 3. 더 많은 데이터를 가진 쪽을 우선 사용
        if dynamic_categories['total_main_categories'] >= file_categories['total_main_categories']:
            logger.info(f"✅ 동적 카테고리 사용: {dynamic_categories['total_main_categories']}개 대분류")
            return dynamic_categories
        else:
            logger.info(f"✅ 파일 기반 카테고리 사용: {file_categories['total_main_categories']}개 대분류")
            return file_categories
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        try:
            count = self.collection.count()
            main_cats = self.main_categories if hasattr(self, 'main_categories') else {}
            
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
                    '2단계 검색 아키텍처 (검색 + 고급 분석)',
                    '고급 의미 분석',
                    '키워드 매칭 최적화',
                    '구조적 관련성 분석',
                    '맥락 인식 검색',
                    '답변 품질 평가',
                    '다중 점수 융합 알고리즘',
                    f'4배 확장된 후보 수집 ({count//4} → {count})',
                    '다양성 보장 중복 제거'
                ]
            }
        except Exception as e:
            logger.error(f"❌ 통계 조회 실패: {e}")
            return {'total_documents': 0, 'is_ready': False}
    
    def rebuild_index(self):
        """인덱스 강제 재구축"""
        logger.info("고급 검색 인덱스 강제 재구축 시작")
        try:
            current_collection_name = getattr(self.collection, 'name', 'company_regulations')
            
            try:
                self.chroma_client.delete_collection(name=current_collection_name)
                logger.info("✅ 기존 컬렉션 삭제")
            except Exception:
                pass
            
            try:
                self.collection = self.chroma_client.create_collection(
                    name="company_regulations",
                    metadata={"description": "고급 검색 기반 회사 내규 벡터 검색 컬렉션"}
                )
                logger.info("✅ 새 컬렉션 생성")
            except Exception:
                import time
                backup_name = f"company_regulations_advanced_{int(time.time())}"
                self.collection = self.chroma_client.create_collection(
                    name=backup_name,
                    metadata={"description": "고급 검색 기반 백업 컬렉션"}
                )
                logger.info(f"✅ 백업 컬렉션 생성: {backup_name}")
            
            return self.build_index(force_rebuild=True)
            
        except Exception as e:
            logger.error(f"❌ 인덱스 재구축 실패: {e}")
            return False

def get_user_friendly_features(use_reranker: bool, context_count: int, results_count: int) -> List[str]:
    """사용자 친화적인 기능 설명 (재랭킹 정보 숨김)"""
    if use_reranker:
        return [
            f"고급 검색으로 {results_count}개 고품질 정보 선별",
            f"총 {context_count}개 컨텍스트 활용",
            "다중 관점 분석으로 정확도 향상",
            "맥락 인식 검색 적용"
        ]
    else:
        return [
            f"향상된 검색으로 {results_count}개 정보 수집",
            f"총 {context_count}개 컨텍스트 활용", 
            "다양성 보장 알고리즘 적용",
            "키워드 매칭 최적화"
        ]

class LLMClient:
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions"):
        """LLM 클라이언트 초기화"""
        self.api_url = api_url
        logger.info(f"LLM 클라이언트 초기화: API URL = {self.api_url}")
    
    def _create_enhanced_system_prompt(self, context: List[Dict[str, Any]], 
                                     conversation_history: List[Dict[str, str]] = None) -> str:
        """일반화된 시스템 프롬프트 생성 (재랭킹 정보 숨김)"""
        
        # 현재 날짜와 시간 정보 가져오기
        try:
            korea_tz = pytz.timezone('Asia/Seoul')
            now = datetime.now(korea_tz)
            current_date = now.strftime("%Y년 %m월 %d일")
            current_time = now.strftime("%H시 %M분")
            current_weekday = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"][now.weekday()]
            datetime_info = f"{current_date} {current_weekday} {current_time}"
        except Exception:
            now = datetime.now()
            datetime_info = now.strftime("%Y년 %m월 %d일 %H시 %M분")
                    
        # 컨텍스트 품질 분석 (내부적으로만)
        high_quality = [c for c in context if c.get('score', 0) >= 0.8]
        medium_quality = [c for c in context if 0.6 <= c.get('score', 0) < 0.8]
        low_quality = [c for c in context if 0.3 <= c.get('score', 0) < 0.6]
        
        system_prompt = f"""당신은 우리 회사의 내규를 정확히 아는 친근한 동료입니다. 직원들의 궁금한 점을 자연스럽지만 정확하게 답변해주세요.

🎯 핵심 원칙 (절대 준수):
• **정확한 정보 활용**: 제공된 {len(context)}개의 고품질 내규 정보를 활용
• **내규 기반 답변 필수**: 반드시 제공된 내규 정보만을 바탕으로 답변
• **추측 금지**: 내규에 명시되지 않은 내용은 절대 추측하지 않기
• **다각도 검토**: 여러 관련 내규를 종합적으로 활용
• **불확실시 명시**: 관련 내규가 없으면 "내규에서 확인이 어려워요"라고 명확히 표현
• **담당 부서 안내**: 정확한 답변이 어려울 때는 경영지원팀 또는 감사팀 문의 안내
• **연차 계산**: 연차일수 = min(기본연차일수 + ⌊ (근속연수(년) − 1) ÷ 2 ⌋, 25)
• **날짜 관련 질문**: 현재 날짜 및 시간 정보를 참고하여 정확한 계산과 안내 제공

💬 자연스러운 대화 방식:
• 친근한 톤으로 설명하되, 내규 내용은 정확히 전달
• 고품질 정보를 우선적으로 활용
• 복잡한 내용은 "쉽게 말하면", "정리하면" 등으로 풀어서 설명
• 이전 대화와 자연스럽게 연결하되, 내규 범위 내에서만
• 날짜나 기간 관련 질문 시 현재 날짜({datetime_info})를 기준으로 정확한 계산 제공

📋 답변 방식:
• 신뢰도가 높은 정보 우선 활용
• 여러 규정이 관련되면 중요도 순으로 정리
• 부분적 정보만 있으면 확실한 부분과 불확실한 부분 구분
• 절대 내규에 없는 내용을 추가하거나 임의 해석 금지
• 날짜 계산이 필요한 경우 현재 날짜({datetime_info})를 기준으로 정확히 계산"""

        if context:
            system_prompt += f"\n\n📚 참고 내규 정보 (총 {len(context)}개, 관련도 순 정렬):\n"
            
            # 현재 검색 결과
            current_contexts = [c for c in context if c.get('source_type') == 'current']
            if current_contexts:
                system_prompt += f"\n[이번 질문 관련 내규: {len(current_contexts)}개]\n"
                for i, item in enumerate(current_contexts, 1):
                    score = item.get('score', 0)
                    # 재랭킹 점수 대신 일반적인 관련도 표현
                    if score >= 0.8:
                        relevance = "매우 관련 높음"
                    elif score >= 0.6:
                        relevance = "관련 높음"
                    else:
                        relevance = "관련 있음"
                    
                    system_prompt += f"{i}. [{item.get('main_category', 'N/A')} > {item.get('sub_category', 'N/A')}]\n"
                    system_prompt += f"   Q: {item.get('question', 'N/A')}\n"
                    system_prompt += f"   A: {item.get('answer', 'N/A')}\n"
                    system_prompt += f"   (관련도: {relevance})\n\n"
                
                system_prompt += f"💡 총 {len(current_contexts)}개의 고품질 내규를 발견했습니다. 관련도가 높은 순서대로 우선 활용하여 정확한 답변을 제공하세요.\n"
            
            # 이전 대화 관련 내규
            previous_contexts = [c for c in context if c.get('source_type') == 'previous']
            if previous_contexts:
                system_prompt += f"\n[이전 대화 관련 내규: {len(previous_contexts)}개]\n"
                for i, item in enumerate(previous_contexts, len(current_contexts) + 1):
                    system_prompt += f"{i}. [{item.get('main_category', 'N/A')} > {item.get('sub_category', 'N/A')}]\n"
                    system_prompt += f"   Q: {item.get('question', 'N/A')}\n"
                    system_prompt += f"   A: {item.get('answer', 'N/A')}\n"
                    system_prompt += f"   (이전 대화 참고)\n\n"
            
            # 품질에 따른 안내 (재랭킹 언급 제거)
            if len(high_quality) == 0 and len(medium_quality) == 0:
                system_prompt += "\n⚠️ 주의: 직접 관련된 내규를 찾기 어려웠습니다. 부분적 정보만 있으므로 확실한 부분만 답변하고 담당 부서 문의를 안내하세요.\n"
            elif len(high_quality) >= 3:
                system_prompt += f"\n✅ 우수: 관련성이 매우 높은 내규 {len(high_quality)}개를 포함하여 총 {len(context)}개의 정확한 정보를 확보했습니다. 이들을 종합적으로 활용하여 완전하고 정확한 답변을 제공하세요.\n"
            else:
                system_prompt += f"\n💡 양호: {len(medium_quality)}개의 관련 내규와 총 {len(context)}개의 정보를 확보했습니다. 관련도가 높은 정보를 우선적으로 활용하세요.\n"
            
            # 일반적인 검색 품질 안내
            system_prompt += f"\n🎯 검색 품질 안내: 제공된 정보는 고급 검색 알고리즘을 통해 분석되어 관련도 순으로 정렬된 고품질 정보입니다. 관련도가 높을수록 더 정확하고 관련성이 높은 정보이므로, 이를 우선적으로 활용해주세요.\n"
        
        # 이전 대화 맥락 추가
        if conversation_history and len(conversation_history) > 0:
            system_prompt += "\n\n💬 이전 대화 맥락:\n"
            recent_history = conversation_history[-3:]
            for msg in recent_history:
                role = "직원" if msg.get('role') == 'user' else "나"
                content = msg.get('content', '')
                truncated_content = content[:150] + ('...' if len(content) > 150 else '')
                system_prompt += f"• {role}: {truncated_content}\n"
            system_prompt += "\n위 대화와 자연스럽게 연결해서 답변해주세요.\n"
        
        system_prompt += "\n\n📞 추가 문의처:\n• 경영지원팀: 인사, 급여, 복리후생, 총무, 재무, 회계 관련\n• 감사팀: 법적 판단, 컴플라이언스, 규정 해석 관련"
        
        return system_prompt
    
    def generate_response_with_history(self, query: str, context: List[Dict[str, Any]], 
                                     conversation_history: List[Dict[str, str]] = None):
        """고급 검색 기반 일반 응답 생성"""
        logger.info(f"📞 고급 검색 기반 응답 생성: '{query[:50]}...' (컨텍스트: {len(context)}개)")
        
        try:
            # 일반화된 시스템 프롬프트 생성
            system_content = self._create_enhanced_system_prompt(context, conversation_history)
            
            messages = [{"role": "system", "content": system_content}]
            
            if conversation_history:
                recent_history = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history
                for msg in recent_history:
                    if msg.get('role') in ['user', 'assistant']:
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
            
            messages.append({"role": "user", "content": query})

            response = requests.post(
                self.api_url,
                json={
                    "model": "qwen3-30b-a3b-mlx",
                    "messages": messages,
                    "temperature": 0.05,  # 정확성 강화
                    "max_tokens": 3000,
                    "top_p": 0.85,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.05,
                    "stream": False
                },
                timeout=150
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['message']['content']
                
                if not generated_text.strip():
                    return "죄송해요, 답변을 생성하지 못했어요."
                
                logger.info(f"✅ 고급 검색 기반 응답 생성 완료 (길이: {len(generated_text)}자)")
                return generated_text
            else:
                return "죄송해요, 지금 시스템에 문제가 있는 것 같아요."
                
        except requests.exceptions.Timeout:
            return "정보 처리에 시간이 걸리고 있어요. 잠시 후 다시 시도해주세요."
        except Exception as e:
            logger.error(f"❌ 응답 생성 실패: {e}", exc_info=True)
            return "죄송해요, 답변 생성 중 문제가 생겼어요."

    def generate_response_stream_with_history(self, query: str, context: List[Dict[str, Any]], 
                                            conversation_history: List[Dict[str, str]] = None):
        """고급 검색 기반 스트리밍 응답 생성 - 수정된 버전"""
        logger.info(f"📞 고급 검색 기반 스트리밍 응답 생성: '{query[:50]}...' (컨텍스트: {len(context)}개)")
        
        try:
            # 일반화된 시스템 프롬프트 생성
            system_content = self._create_enhanced_system_prompt(context, conversation_history)
            
            messages = [{"role": "system", "content": system_content}]
            
            if conversation_history:
                recent_history = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history
                for msg in recent_history:
                    if msg.get('role') in ['user', 'assistant']:
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
            
            messages.append({"role": "user", "content": query})

            response = requests.post(
                self.api_url,
                json={
                    "model": "qwen3-30b-a3b-mlx",
                    "messages": messages,
                    "temperature": 0.05,  # 정확성 강화
                    "max_tokens": 3000,
                    "top_p": 0.85,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.05,
                    "stream": True
                },
                timeout=150,
                stream=True
            )
            
            if response.status_code == 200:
                generated_text = ""
                
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data_str = line_text[6:]
                            
                            if data_str.strip() == '[DONE]':
                                break
                                
                            try:
                                data = json.loads(data_str)
                                delta = data['choices'][0]['delta']
                                
                                if 'content' in delta:
                                    content = delta['content']
                                    generated_text += content
                                    # 딕셔너리 형태로 yield
                                    yield {
                                        "type": "content",
                                        "content": content
                                    }
                                
                                if 'finish_reason' in data['choices'][0] and data['choices'][0]['finish_reason']:
                                    yield {
                                        "type": "finish",
                                        "content": ""
                                    }
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                            except (KeyError, IndexError):
                                continue
                
                if not generated_text.strip():
                    yield {
                        "type": "error",
                        "content": "죄송해요, 답변을 생성하지 못했어요."
                    }
                
                logger.info(f"✅ 고급 검색 기반 스트리밍 응답 생성 완료 (길이: {len(generated_text)}자)")
            else:
                yield {
                    "type": "error",
                    "content": "죄송해요, 지금 시스템에 문제가 있는 것 같아요."
                }
                
        except requests.exceptions.Timeout:
            yield {
                "type": "error",
                "content": "정보 처리에 시간이 걸리고 있어요. 잠시 후 다시 시도해주세요."
            }
        except Exception as e:
            logger.error(f"❌ 스트리밍 응답 생성 실패: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": "죄송해요, 답변 생성 중 문제가 생겼어요."
            }

# FastAPI 앱 설정
app = FastAPI(
    title="고품질 검색 기반 회사 내규 RAG 시스템 (동적 카테고리 지원)",
    description="""
    **FastAPI 기반 고품질 검색 회사 내규 RAG 시스템 (동적 카테고리 지원)**
    
    ## 주요 특징
    - 🧠 **고급 검색 알고리즘**: 다중 관점 분석으로 정확한 정보 제공
    - 🎯 **2단계 검색 아키텍처**: 광범위한 후보 수집 → 정교한 품질 분석
    - 🤖 **다중 검색 방법**: 고급 검색 또는 향상된 검색 선택 가능
    - 📊 **다중 점수 융합**: 의미적 유사도 + 키워드 매칭 + 구조적 관련성
    - 🧠 **맥락 인식 검색**: 이전 대화를 고려한 동적 점수 조정
    - 📝 **키워드 매칭 최적화**: 전통적 정보 검색 기법 활용
    - 🏗️ **구조적 관련성 분석**: 질문-답변 구조 고려
    - 💎 **답변 품질 평가**: 길이, 구조, 예시 포함 여부 등 종합 평가
    - 🔄 **동적 카테고리 관리**: ChromaDB에서 실시간 카테고리 조회 (신규 데이터 즉시 반영)
    
    ## 검색 알고리즘
    - **의미적 유사도**: 30% (고급 의미 분석)
    - **구조적 관련성**: 25% (질문-답변 매칭)
    - **키워드 관련성**: 20% (키워드 매칭)
    - **원본 벡터 점수**: 15% (기본 유사도)
    - **답변 품질**: 5% (구조적 완성도)
    - **맥락 관련성**: 5% (이전 대화 고려)
    
    ## 동적 카테고리 기능
    - **실시간 카테고리 조회**: ChromaDB에서 직접 카테고리 정보 추출
    - **신규 데이터 즉시 반영**: 새로 추가된 내규가 바로 카테고리에 표시
    - **기존 데이터 호환**: 파일 기반 데이터와 벡터 DB 데이터 통합 관리
    - **자동 fallback**: ChromaDB 조회 실패 시 기존 방식으로 자동 전환
    
    ## 성능 지표
    - **후보 수집**: 4배 확장 (top_k × 4개 후보)
    - **검색 정확도**: 고급 알고리즘으로 최고 수준
    - **다양성 보장**: 중복 제거 + 품질 유지
    - **맥락 인식**: 이전 대화 맥락 반영
    - **카테고리 관리**: 동적 + 정적 하이브리드 방식
    """,
    version="4.2.0-dynamic-categories",
    contact={
        "name": "고품질 검색 RAG 시스템 개발팀 (동적 카테고리 지원)",
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

# 전역 예외 처리기
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"❌ 요청 검증 실패: {exc}")
    
    error_details = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        error_details.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "요청 데이터 검증 실패",
            "details": error_details,
            "help": {
                "message": "API 문서를 참조하여 올바른 형식으로 요청해주세요.",
                "docs_url": "/docs",
                "search_features": [
                    "고급 검색으로 정확도 대폭 향상",
                    "2단계 검색 아키텍처 적용",
                    "다중 점수 융합 알고리즘",
                    "맥락 인식 검색",
                    "동적 카테고리 관리"
                ]
            }
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    logger.error(f"❌ 내부 서버 오류: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "내부 서버 오류가 발생했습니다.",
            "message": "시스템 서버 로그를 확인해주세요.",
            "request_id": id(request)
        }
    )

# 전역 객체들
rag_system: CompanyRegulationsRAGSystem = None
llm_client: LLMClient = None

def initialize_system():
    """고급 검색 기반 시스템 초기화"""
    global rag_system, llm_client
    
    if rag_system is not None and llm_client is not None:
        logger.info("시스템이 이미 초기화되어 있습니다.")
        return

    logger.info("🎯 고급 검색 기반 RAG 시스템 초기화 시작... (동적 카테고리 지원)")
    
    try:
        rag_system = CompanyRegulationsRAGSystem()
        llm_client = LLMClient()
        
        data_directory = "./data_json"
        
        if os.path.exists(data_directory):
            logger.info("회사 내규 데이터 로드 및 인덱스 구축...")
            if rag_system.load_company_regulations_data(data_directory):
                if rag_system.build_index():
                    stats = rag_system.get_stats()
                    logger.info(f"✅ 고급 검색 시스템 초기화 완료 (동적 카테고리 지원): {stats['total_documents']}개 청크")
                    logger.info(f"📊 청킹 통계: {stats.get('chunk_statistics', {})}")
                    logger.info(f"🎯 고급 기능: {', '.join(stats.get('enhanced_features', [])[:3])}")
                else:
                    logger.error("❌ 인덱스 구축 실패")
            else:
                logger.error("❌ 내규 데이터 로드 실패")
        else:
            logger.warning(f"⚠️ 데이터 디렉토리 없음: {data_directory} (동적 카테고리만 사용)")

    except Exception as e:
        logger.critical(f"🔥 시스템 초기화 실패: {e}", exc_info=True)
        rag_system = None
        llm_client = None

@app.get("/api/health", response_model=HealthResponse, summary="시스템 상태 확인", description="고품질 검색 기반 RAG 시스템의 상태와 통계를 확인합니다 (동적 카테고리 지원).")
async def health_check():
    """시스템 헬스 체크"""
    if rag_system is None or llm_client is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "initializing_or_error",
                "rag_ready": False,
                "regulations_count": 0,
                "main_categories_count": 0,
                "improvements": "고급 검색으로 정확도 대폭 향상 + 동적 카테고리 지원",
                "message": "RAG 시스템 초기화 중이거나 오류가 발생했습니다."
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
        improvements="고급 검색으로 정확도 대폭 향상 + 동적 카테고리 지원",
        enhanced_features=stats.get('enhanced_features', []) + ["동적 카테고리 관리", "실시간 데이터 반영"]
    )

@app.get("/api/categories", summary="카테고리 정보 조회 (동적)", description="동적 카테고리 조회: ChromaDB에서 실시간으로 카테고리 정보를 조회하여 새로 추가된 데이터도 즉시 반영합니다.")
async def get_categories():
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    categories_info = rag_system.get_categories()
    
    # 추가적으로 리스트 형태로도 제공 (자연스러운 숫자 정렬된 순서로)
    sorted_categories_list = []
    for main_category, info in categories_info['main_categories'].items():
        sorted_categories_list.append({
            'main_category': main_category,
            'file_name': info['file_name'],
            'sub_categories': info['sub_categories'],
            'total_faqs': info['total_faqs']
        })
    
    return {
        "categories": categories_info,
        "sorted_categories_list": sorted_categories_list,  # 자연스러운 숫자 정렬된 리스트 형태 추가
        "database": "ChromaDB (Advanced Search + Dynamic Categories)",
        "system_type": "고품질 검색 기반 회사 내규 시스템 (동적 카테고리 지원)",
        "data_source": "ChromaDB 우선 + 파일 fallback",
        "sorting_info": {
            "type": "natural_numeric_sorting",
            "description": "1-1, 1-2, 1-3, ..., 1-9, 1-10, 1-11, 1-12 순서로 정렬됩니다"
        }
    }

@app.get("/api/categories_dynamic", summary="동적 카테고리 정보 조회 (ChromaDB 직접)", 
         description="ChromaDB에서 직접 카테고리 정보를 동적으로 조회합니다 (새로 추가된 데이터 포함).")
async def get_categories_dynamic():
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    categories_info = rag_system.get_categories_from_vector_db()
    
    # 추가적으로 리스트 형태로도 제공
    sorted_categories_list = []
    for main_category, info in categories_info['main_categories'].items():
        sorted_categories_list.append({
            'main_category': main_category,
            'file_name': info['file_name'],
            'sub_categories': info['sub_categories'],
            'total_faqs': info['total_faqs']
        })
    
    return {
        "categories": categories_info,
        "sorted_categories_list": sorted_categories_list,
        "database": "ChromaDB (Dynamic Search)",
        "system_type": "동적 카테고리 조회 시스템",
        "data_source": "ChromaDB 직접 조회 (실시간)",
        "sorting_info": {
            "type": "natural_numeric_sorting",
            "description": "1-1, 1-2, 1-3, ..., 1-9, 1-10, 1-11, 1-12 순서로 정렬됩니다"
        }
    }

@app.post("/api/refresh_categories", summary="카테고리 캐시 새로고침",
          description="ChromaDB에서 최신 카테고리 정보를 다시 로드합니다.")
async def refresh_categories():
    """카테고리 정보 새로고침"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    try:
        # ChromaDB에서 최신 카테고리 정보 조회
        dynamic_categories = rag_system.get_categories_from_vector_db()
        
        return {
            "success": True,
            "message": "카테고리 정보가 성공적으로 새로고침되었습니다.",
            "updated_categories": dynamic_categories['total_main_categories'],
            "total_regulations": dynamic_categories['total_regulations'],
            "main_categories": list(dynamic_categories['main_categories'].keys()),
            "refreshed_at": datetime.now(pytz.timezone('Asia/Seoul')).isoformat(),
            "data_source": "ChromaDB 직접 조회"
        }
        
    except Exception as e:
        logger.error(f"❌ 카테고리 새로고침 실패: {e}")
        raise HTTPException(status_code=500, detail=f"카테고리 새로고침 실패: {str(e)}")

@app.post("/api/search", response_model=SearchResponse, summary="고품질 내규 검색", description="고급 검색 알고리즘으로 정확한 관련 내규를 찾습니다.")
async def search_regulations(request: SearchRequest):
    """고품질 회사 내규 검색"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    # 고급 검색 사용
    results = rag_system.search_with_advanced_rerank(
        request.query, 
        request.top_k,
        request.main_category_filter,
        conversation_history=None,
        min_relevance_score=0.2
    )
    
    # rerank_details 제거
    clean_results = []
    for result in results:
        clean_result = {k: v for k, v in result.items() if k != 'rerank_details'}
        clean_results.append(SearchResult(**clean_result))
    
    return SearchResponse(
        query=request.query,
        results=clean_results,
        count=len(results),
        main_category_filter=request.main_category_filter,
        search_type="고급 검색 기반 2단계 검색",
        min_relevance=0.2,
        enhanced_features=[
            "고급 의미 분석",
            "키워드 매칭 최적화",
            "구조적 관련성 분석",
            "다중 점수 융합 알고리즘",
            f"4배 확장된 후보 수집 (최대 {request.top_k * 4}개)"
        ]
    )

@app.post("/api/chat", response_model=ChatResponse, summary="내규 상담", description="고품질 검색을 통한 내규 상담을 제공합니다.")
async def chat_with_rag(request: ChatRequest):
    """내규 상담 (재랭킹 정보 숨김)"""
    try:
        if not rag_system or not llm_client:
            raise HTTPException(status_code=503, detail="시스템이 초기화되지 않았습니다.")
        
        # conversation_history를 Dict 형식으로 변환
        conversation_history = []
        for msg in request.conversation_history:
            conversation_history.append({
                "role": msg.role,
                "content": msg.content,
                "context": getattr(msg, 'context', None)
            })
        
        # 검색 (내부적으로는 재랭킹 사용하되 사용자에게는 숨김)
        if request.use_reranker:
            current_results = rag_system.search_with_advanced_rerank(
                request.query, 
                top_k=20,
                main_category_filter=request.main_category_filter,
                conversation_history=conversation_history,
                min_relevance_score=0.2,
                reranker_type=request.reranker_type.value
            )
            search_type = "고급 검색"
            response_type = "고품질 정확성 강화 대화형"
        else:
            current_results = rag_system.search_with_enhanced_retrieval(
                request.query, 
                top_k=20,
                main_category_filter=request.main_category_filter,
                min_relevance_score=0.3
            )
            search_type = "향상된 검색"
            response_type = "향상된 대화형"
        
        # 컨텍스트 결합
        combined_context = rag_system.combine_contexts_with_history(
            current_results, 
            conversation_history, 
            max_total_context=25
        )
        
        # 응답 생성
        response = llm_client.generate_response_with_history(
            request.query, 
            combined_context, 
            conversation_history
        )
        
        # 품질 분석 (내부적으로)
        if request.use_reranker:
            high_quality = len([c for c in combined_context if c.get('score', 0) >= 0.8])
            medium_quality = len([c for c in combined_context if 0.6 <= c.get('score', 0) < 0.8])
            low_quality = len([c for c in combined_context if 0.3 <= c.get('score', 0) < 0.6])
        else:
            high_quality = len([c for c in combined_context if c.get('score', 0) >= 0.7])
            medium_quality = len([c for c in combined_context if 0.5 <= c.get('score', 0) < 0.7])
            low_quality = len([c for c in combined_context if 0.3 <= c.get('score', 0) < 0.5])
        
        # SearchResult 생성 시 rerank_details 제거
        context_results = []
        for ctx in combined_context:
            clean_ctx = {k: v for k, v in ctx.items() if k != 'rerank_details'}
            context_results.append(SearchResult(**clean_ctx))
        
        # 사용자 친화적인 기능 설명
        enhanced_features = get_user_friendly_features(
            request.use_reranker, 
            len(combined_context), 
            len(current_results)
        )
        
        return ChatResponse(
            query=request.query,
            response=response,
            context=context_results,
            context_count=len(combined_context),
            context_quality={
                "high_relevance": high_quality,
                "medium_relevance": medium_quality,
                "low_relevance": low_quality
            },
            search_type=search_type,
            response_type=response_type,
            enhanced_features=enhanced_features
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 상담 처리 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"상담 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/api/chat_stream", summary="실시간 내규 상담", description="고품질 검색을 통한 실시간 스트리밍 내규 상담을 제공합니다.")
async def chat_with_rag_stream(request: StreamChatRequest):
    """스트리밍 상담 (재랭킹 정보 숨김)"""
    try:
        if not rag_system or not llm_client:
            raise HTTPException(status_code=503, detail="시스템이 초기화되지 않았습니다.")
        
        conversation_history = request.conversation_history or []
        
        # 검색 (내부적으로는 재랭킹 사용하되 사용자에게는 숨김)
        if request.use_reranker:
            current_results = rag_system.search_with_advanced_rerank(
                request.query, 
                top_k=20,
                main_category_filter=request.main_category_filter,
                conversation_history=conversation_history,
                min_relevance_score=0.2,
                reranker_type=request.reranker_type.value
            )
            search_type = "고급 검색"
        else:
            current_results = rag_system.search_with_enhanced_retrieval(
                request.query, 
                top_k=20,
                main_category_filter=request.main_category_filter,
                min_relevance_score=0.3
            )
            search_type = "향상된 검색"
        
        # 컨텍스트 결합
        combined_context = rag_system.combine_contexts_with_history(
            current_results, 
            conversation_history, 
            max_total_context=25
        )
        
        def generate():
            try:
                # 품질 분석 (내부적으로)
                if request.use_reranker:
                    high_quality = len([c for c in combined_context if c.get('score', 0) >= 0.8])
                    medium_quality = len([c for c in combined_context if 0.6 <= c.get('score', 0) < 0.8])
                    low_quality = len([c for c in combined_context if 0.3 <= c.get('score', 0) < 0.6])
                else:
                    high_quality = len([c for c in combined_context if c.get('score', 0) >= 0.7])
                    medium_quality = len([c for c in combined_context if 0.5 <= c.get('score', 0) < 0.7])
                    low_quality = len([c for c in combined_context if 0.3 <= c.get('score', 0) < 0.5])
                
                # 사용자 친화적인 기능 설명
                enhanced_features = get_user_friendly_features(
                    request.use_reranker, 
                    len(combined_context), 
                    len(current_results)
                )
                
                # rerank_details 제거
                clean_combined_context = []
                for ctx in combined_context:
                    clean_ctx = {k: v for k, v in ctx.items() if k != 'rerank_details'}
                    clean_combined_context.append(clean_ctx)
                
                context_data = {
                    "type": "context",
                    "query": request.query,
                    "context": clean_combined_context,
                    "context_count": len(combined_context),
                    "context_quality": {
                        "high_relevance": high_quality,
                        "medium_relevance": medium_quality,
                        "low_relevance": low_quality
                    },
                    "search_type": search_type,
                    "enhanced_features": enhanced_features
                }
                
                yield f"data: {json.dumps(context_data, ensure_ascii=False)}\n\n"
                
                # 스트리밍 응답 생성 - 수정된 부분
                for chunk in llm_client.generate_response_stream_with_history(
                    request.query, 
                    combined_context, 
                    conversation_history
                ):
                    # chunk는 이제 딕셔너리이므로 직접 사용 가능
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
                yield f"data: {json.dumps({'type': 'stream_end'}, ensure_ascii=False)}\n\n"
                
            except Exception as stream_error:
                logger.error(f"❌ 스트리밍 오류: {stream_error}", exc_info=True)
                error_data = {
                    "type": "error",
                    "content": f"스트리밍 중 오류가 발생했습니다: {str(stream_error)}"
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 스트리밍 상담 초기화 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"스트리밍 상담 초기화 중 오류: {str(e)}"
        )

@app.post("/api/rebuild_index", summary="인덱스 재구축", description="내규 데이터를 다시 로드하고 인덱스를 재구축합니다.")
async def rebuild_index():
    """인덱스 재구축"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    if rag_system.load_company_regulations_data("./data_json"):
        if rag_system.rebuild_index():
            stats = rag_system.get_stats()
            return {
                "message": "인덱스가 성공적으로 재구축되었습니다",
                "regulations_count": stats['total_documents'],
                "chunk_statistics": stats.get('chunk_statistics', {}),
                "improvements": "고급 검색으로 정확도 대폭 향상 + 동적 카테고리 지원",
                "enhanced_features": stats.get('enhanced_features', []),
                "search_performance": {
                    "retrieval_expansion": "4배 후보 수집",
                    "quality_algorithm": "다중 점수 융합",
                    "accuracy_boost": "고급 의미 분석",
                    "context_awareness": "이전 대화 맥락 반영",
                    "dynamic_categories": "실시간 카테고리 관리"
                }
            }
        else:
            raise HTTPException(status_code=500, detail="인덱스 재구축 실패")
    else:
        raise HTTPException(status_code=500, detail="데이터 로드 실패")

@app.get("/api/stats", summary="시스템 통계", description="고품질 검색 시스템의 상세 통계와 성능 지표를 확인합니다.")
async def get_stats():
    """시스템 통계 정보 (재랭킹 정보 일반화)"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    stats = rag_system.get_stats()
    stats.update({
        "system_type": "고품질 회사 내규 검색 시스템 (동적 카테고리 지원)",
        "search_features": [
            "고급 검색 알고리즘",
            "다중 관점 분석",
            "맥락 인식 검색",
            "구조적 관련성 분석",
            "키워드 매칭 최적화",
            "답변 품질 평가",
            "다양성 보장 알고리즘",
            "동적 카테고리 관리"
        ],
        "quality_metrics": {
            "accuracy_enhancement": "고급 알고리즘 적용",
            "context_awareness": "이전 대화 고려",
            "relevance_scoring": "다중 점수 융합",
            "diversity_guarantee": "중복 제거 보장",
            "dynamic_categories": "실시간 카테고리 반영"
        },
        "performance_metrics": {
            "search_expansion": "확장된 후보 수집",
            "quality_algorithm": "다중 관점 분석",
            "accuracy_improvement": "구조적 + 키워드 + 의미적",
            "context_window": 25,
            "response_temperature": 0.05,
            "min_relevance": "동적 조정",
            "category_management": "ChromaDB 우선 + 파일 fallback"
        }
    })
    
    # 재랭킹 관련 기술적 내용 제거
    if 'enhanced_features' in stats:
        stats['enhanced_features'] = [
            feature.replace('재랭킹', '고급 검색').replace('Rerank', '고급 검색')
            .replace('Cross-encoder', '고급 알고리즘').replace('TF-IDF', '키워드 매칭')
            for feature in stats['enhanced_features']
        ] + ["동적 카테고리 관리", "실시간 데이터 반영"]
    
    return stats

# 간단한 테스트 엔드포인트
@app.post("/api/test/simple_chat", summary="채팅 테스트", description="고품질 검색을 통한 채팅 기능을 테스트합니다.")
async def simple_chat_test(request: SimpleTestRequest):
    """간단한 채팅 테스트 (재랭킹 정보 숨김)"""
    try:
        if not rag_system or not llm_client:
            raise HTTPException(status_code=503, detail="시스템이 초기화되지 않았습니다.")
        
        # 검색 (내부적으로는 재랭킹 사용)
        if request.use_reranker:
            results = rag_system.search_with_advanced_rerank(
                request.query, 
                top_k=15, 
                main_category_filter=request.category,
                conversation_history=None,
                min_relevance_score=0.2,
                reranker_type=request.reranker_type.value
            )
            search_method = "고급 검색"
        else:
            results = rag_system.search_with_enhanced_retrieval(
                request.query, 
                top_k=15, 
                main_category_filter=request.category,
                min_relevance_score=0.3
            )
            search_method = "향상된 검색"
        
        if results:
            response = llm_client.generate_response_with_history(request.query, results, [])
            
            # 일반적인 분석 정보만 제공
            analysis_info = {
                "total_candidates": len(results),
                "avg_score": sum(r['score'] for r in results) / len(results) if results else 0,
                "search_method": search_method,
                "high_quality_count": len([r for r in results if r.get('score', 0) >= 0.7])
            }
            
            info_message = f"{search_method}으로 {len(results)}개 고품질 정보 제공"
            
            return {
                "query": request.query,
                "response": response,
                "found_results": len(results),
                "status": "success",
                "performance": analysis_info,
                "info": info_message
            }
        else:
            return {
                "query": request.query,
                "response": f"관련된 내규를 찾을 수 없습니다. ({search_method} 기준)",
                "found_results": 0,
                "status": "no_results",
                "search_info": f"{search_method} 최소 점수 미만"
            }
            
    except Exception as e:
        logger.error(f"❌ 채팅 테스트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"테스트 실패: {str(e)}")

@app.get("/api/test/stream_simple", summary="스트리밍 테스트", description="고품질 검색을 이용한 간단한 스트리밍 테스트")
async def simple_stream_test(
    query: str = Query("휴가 신청 방법", description="검색할 질의", example="휴가 신청 방법"),
    use_reranker: bool = Query(True, description="고급 검색 사용 여부", example=True),
    reranker_type: RerankerType = Query(RerankerType.qwen3, description="검색 방법 선택")
):
    """간단한 스트리밍 테스트"""
    try:
        def generate():
            try:
                search_method = "고급 검색" if use_reranker else "향상된 검색"
                yield f"data: {json.dumps({'type': 'start', 'message': f'{search_method} 질문 처리 중: {query}'}, ensure_ascii=False)}\n\n"
                
                if not rag_system or not llm_client:
                    yield f"data: {json.dumps({'type': 'error', 'content': '시스템이 초기화되지 않았습니다.'}, ensure_ascii=False)}\n\n"
                    return
                
                # 검색
                if use_reranker:
                    results = rag_system.search_with_advanced_rerank(
                        query, 
                        top_k=10,
                        conversation_history=None,
                        min_relevance_score=0.2,
                        reranker_type=reranker_type.value
                    )
                else:
                    results = rag_system.search_with_enhanced_retrieval(
                        query, 
                        top_k=10,
                        min_relevance_score=0.3
                    )
                
                # 결과 분석
                analysis_info = {
                    "total_results": len(results),
                    "avg_score": sum(r['score'] for r in results) / len(results) if results else 0,
                    "search_method": search_method,
                    "high_quality_count": len([r for r in results if r.get('score', 0) >= 0.7])
                }
                
                avg_score = analysis_info["avg_score"]
                info_message = f'{search_method} 완료: {len(results)}개 결과, 평균 점수 {avg_score:.3f}'
                yield f"data: {json.dumps({'type': 'info', 'message': info_message}, ensure_ascii=False)}\n\n"
                
                # 스트리밍 응답 - 수정된 부분
                for chunk in llm_client.generate_response_stream_with_history(query, results, []):
                    # chunk는 이제 딕셔너리이므로 직접 사용 가능
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    
                yield f"data: {json.dumps({'type': 'analysis_summary', 'content': analysis_info}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'stream_end'}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logger.error(f"❌ 스트리밍 테스트 오류: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': f'오류: {str(e)}'}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache'}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스트리밍 테스트 실패: {str(e)}")

@app.get("/api/test/search_demo", summary="검색 성능 데모", description="고급 검색 알고리즘의 성능을 시연합니다.")
async def search_demo(
    query: str = Query("연차 휴가", description="검색할 질의", example="연차 휴가"), 
    top_k: int = Query(5, description="반환할 결과 수", example=5, ge=1, le=20),
    reranker_type: RerankerType = Query(RerankerType.qwen3, description="검색 방법 선택")
):
    """검색 성능 데모"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="시스템이 초기화되지 않았습니다.")
        
        # 고급 검색
        results = rag_system.search_with_advanced_rerank(
            query, 
            top_k=top_k,
            conversation_history=None,
            min_relevance_score=0.1,  # 낮은 임계값으로 더 많은 결과 보기
            reranker_type=reranker_type.value
        )
        
        demo_results = []
        for i, result in enumerate(results):
            # 내부 정보 숨김 처리
            demo_results.append({
                "rank": i + 1,
                "question": result.get('question', ''),
                "category": f"{result.get('main_category', '')} > {result.get('sub_category', '')}",
                "final_score": result.get('score', 0),
                "relevance_level": "매우 높음" if result.get('score', 0) >= 0.8 else "높음" if result.get('score', 0) >= 0.6 else "보통",
                "chunk_type": result.get('chunk_type', ''),
                "answer_preview": result.get('answer', '')[:100] + "..." if len(result.get('answer', '')) > 100 else result.get('answer', '')
            })
        
        return {
            "query": query,
            "search_method": "고급 검색",
            "search_algorithm": {
                "name": "다중 점수 융합 고급 검색",
                "description": "의미적 분석, 키워드 매칭, 구조적 관련성을 종합한 고품질 검색",
                "features": [
                    "고급 의미 분석",
                    "키워드 매칭 최적화",
                    "질문-답변 구조 분석",
                    "답변 품질 평가",
                    "다양성 보장 알고리즘"
                ]
            },
            "results": demo_results,
            "performance_summary": {
                "total_results": len(results),
                "avg_score": sum(r['score'] for r in results) / len(results) if results else 0,
                "high_quality_count": len([r for r in results if r.get('score', 0) >= 0.8]),
                "score_distribution": {
                    "excellent": len([r for r in results if r.get('score', 0) >= 0.8]),
                    "good": len([r for r in results if 0.6 <= r.get('score', 0) < 0.8]),
                    "fair": len([r for r in results if 0.4 <= r.get('score', 0) < 0.6]),
                    "poor": len([r for r in results if r.get('score', 0) < 0.4])
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 검색 데모 실패: {e}")
        raise HTTPException(status_code=500, detail=f"검색 데모 실패: {str(e)}")

# 사용자 정의 docs 설정
@app.get("/api/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/api/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.0/bundles/redoc.standalone.js",
    )

# 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    initialize_system()

if __name__ == '__main__':
    print("=" * 90)
    print("🎯 FastAPI 기반 고품질 검색 회사 내규 RAG 서버 (동적 카테고리 지원)")
    print("=" * 90)
    print("🚀 다중 검색 방법:")
    print("   🧠 고급 검색: 다중 관점 분석으로 최고 정확도 (기본)")
    print("   📊 향상된 검색: 기존 알고리즘 개선 버전")
    print("   ⚙️ 옵션 선택 가능: 고급 검색 사용/미사용 선택")
    print("=" * 90)
    print("🔧 검색 방법 비교:")
    print("   고급 검색        : 다중 점수 융합, 매우 높은 정확도, 보통 속도 (기본)")
    print("   향상된 검색      : 키워드 매칭, 높은 정확도, 빠른 속도")
    print("=" * 90)
    print("📊 고급 검색 알고리즘 가중치:")
    print("   의미적 유사도     : 30% (고급 의미 분석)")
    print("   구조적 관련성     : 25% (질문-답변 매칭)")
    print("   키워드 관련성     : 20% (키워드 매칭)")
    print("   원본 벡터 점수    : 15% (기본 유사도)")
    print("   답변 품질        : 5%  (구조적 완성도)")
    print("   맥락 관련성      : 5%  (이전 대화 고려)")
    print("=" * 90)
    print("🔄 동적 카테고리 관리:")
    print("   • ChromaDB 우선 조회: 새로 추가된 데이터 즉시 반영")
    print("   • 파일 기반 fallback: ChromaDB 조회 실패 시 자동 전환")  
    print("   • 실시간 업데이트: 벡터 DB 추가 시 카테고리 자동 업데이트")
    print("   • 자연스러운 정렬: 1-1, 1-2, ..., 1-10, 1-11 순서")
    print("=" * 90)
    print("⚙️ 사용 옵션:")
    print("   use_reranker=true + reranker_type='qwen3'       : 고급 검색 (기본)")
    print("   use_reranker=true + reranker_type='llm_api'     : 고급 검색 (LLM)")
    print("   use_reranker=true + reranker_type='sentence_transformer': 고급 검색 (벡터)")
    print("   use_reranker=false                              : 향상된 검색")
    print("=" * 90)
    print("🔧 FastAPI 엔드포인트:")
    print("   - GET  /api/health              : 시스템 상태 확인")
    print("   - GET  /api/categories          : 카테고리 조회 (동적 + fallback)")
    print("   - GET  /api/categories_dynamic  : 동적 카테고리 조회 (ChromaDB 직접)")
    print("   - POST /api/refresh_categories  : 카테고리 캐시 새로고침")
    print("   - POST /api/search              : 고품질 검색")
    print("   - POST /api/chat                : 다중 검색 방법 상담")
    print("   - POST /api/chat_stream         : 다중 검색 방법 실시간 상담 ⚡")
    print("   - POST /api/rebuild_index       : 인덱스 재구축")
    print("   - GET  /api/stats               : 성능 지표 포함 통계")
    print("   - POST /api/test/simple_chat    : 🧪 다중 검색 방법 테스트")
    print("   - GET  /api/test/stream_simple  : 🧪 다중 검색 스트리밍 테스트")
    print("   - GET  /api/test/search_demo    : 🎯 검색 성능 데모 (점수 분석)")
    print("   - GET  /docs                    : 🎯 Swagger UI 문서 (API 테스트) 🎯")
    print("=" * 90)
    print("📖 API 문서 및 테스트:")
    print("   http://localhost:5000/docs  ← 🎯 고품질 검색 API 테스트하세요!")
    print("=" * 90)
    print("🧪 다중 검색 테스트 방법:")
    print("   1. 고급 검색 (기본, 매우 높은 정확도):")
    print("      {\"query\": \"휴가신청\", \"use_reranker\": true, \"reranker_type\": \"qwen3\"}")
    print("   2. 고급 검색 (LLM 방식):")
    print("      {\"query\": \"휴가신청\", \"use_reranker\": true, \"reranker_type\": \"llm_api\"}")
    print("   3. 고급 검색 (벡터 방식):")
    print("      {\"query\": \"휴가신청\", \"use_reranker\": true, \"reranker_type\": \"sentence_transformer\"}")
    print("   4. 향상된 검색 (가장 빠름):")
    print("      {\"query\": \"휴가신청\", \"use_reranker\": false}")
    print("=" * 90)
    print("📝 다중 검색 예제:")
    print("   # 고급 검색 (기본)")
    print("   curl -X POST 'http://localhost:5000/api/test/simple_chat' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"query\": \"휴가신청\", \"reranker_type\": \"qwen3\"}'")
    print("   # 향상된 검색")
    print("   curl -X POST 'http://localhost:5000/api/test/simple_chat' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"query\": \"휴가신청\", \"use_reranker\": false}'")
    print("=" * 90)
    print("🎯 검색 방법 선택 가이드:")
    print("   🧠 고급 검색      : 최고 정확도, 다중 관점 분석, 안정적 성능 (기본)")
    print("   📊 향상된 검색    : 빠른 속도, 키워드 매칭, 간단한 질문에 적합")
    print("=" * 90)
    print("🔍 처리 과정:")
    print("   고급 검색: 4x 후보수집 → 의미분석 → 6가지점수융합 → 상위선별")
    print("   향상된 검색: 3x 후보수집 → 키워드매칭 → 점수조정 → 상위선별")
    print("=" * 90)
    print("🔄 동적 카테고리 워크플로우:")
    print("   1. 포트 5001에서 새 내규 벡터 DB 추가")
    print("   2. 포트 5000의 /api/categories 자동으로 새 데이터 인식")
    print("   3. 재시작 없이 즉시 카테고리 목록에 반영")
    print("   4. ChromaDB 우선, 파일 기반 fallback으로 안정성 보장")
    print("=" * 90)
    print("⚠️ 중요 사항:")
    print("   • 고급 검색: 다중 점수 융합으로 최고 정확도")
    print("   • 향상된 검색: 키워드 매칭 기반으로 빠른 속도")
    print("   • 동적 카테고리: 새 데이터 추가 시 즉시 반영")
    print("   • 사용자에게는 기술적 세부사항 숨김 처리")
    print("=" * 90)
    
    # 시스템 초기화
    initialize_system() 
    
    logger.info("🎯 고품질 검색 FastAPI 서버 시작 중... (동적 카테고리 지원)")
    
    # FastAPI 앱 실행
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level="info")