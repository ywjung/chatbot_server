import json
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic 모델 정의
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

class CompanyRegulationsRAGSystem:
    def __init__(self, model_name: str = "nlpai-lab/KURE-v1", persist_directory: str = "./chroma_db"):
        """
        개선된 ChromaDB 기반 회사 전체 내규 RAG 시스템 초기화 (향상된 검색 및 정보량)

        Args:
            model_name: HuggingFace 모델 이름 또는 로컬 디렉토리 이름
            persist_directory: ChromaDB 저장 디렉토리
        """
        model_path = os.path.join("./models", model_name.replace("/", "-"))

        # 로컬 디렉토리에서 모델 로드 또는 다운로드
        if not os.path.exists(model_path):
            logger.info(f"📦 로컬에 모델이 존재하지 않습니다. Hugging Face에서 '{model_name}' 모델을 다운로드합니다...")
            try:
                model = SentenceTransformer(model_name)
                model.save(model_path)
                logger.info(f"✅ 모델 다운로드 및 저장 완료: {model_path}")
            except Exception as e:
                logger.error(f"❌ 모델 다운로드 실패: {e}", exc_info=True)
                raise
        else:
            logger.info(f"🔄 기존 모델 로드: {model_path}")

        # 저장된 모델 로드
        self.model = SentenceTransformer(model_path)
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

    def _clear_collection_safely(self):
        """컬렉션 데이터를 안전하게 삭제"""
        try:
            # 방법 1: 모든 ID를 가져와서 개별 삭제 시도
            try:
                logger.info("🔄 방법 1: 개별 ID 삭제 시도...")
                all_results = self.collection.get()
                if all_results and 'ids' in all_results and all_results['ids']:
                    batch_size = 1000
                    total_ids = all_results['ids']
                    
                    for i in range(0, len(total_ids), batch_size):
                        batch_ids = total_ids[i:i+batch_size]
                        self.collection.delete(ids=batch_ids)
                        logger.info(f"   삭제 배치 {i//batch_size + 1}: {len(batch_ids)}개 ID")
                    
                    logger.info("✅ 개별 ID 삭제 완료")
                    return True
            except Exception as e:
                logger.warning(f"⚠️ 개별 ID 삭제 실패: {e}")
            
            # 방법 2: 컬렉션 재생성
            try:
                logger.info("🔄 방법 2: 컬렉션 재생성 시도...")
                collection_name = self.collection.name
                collection_metadata = self.collection.metadata
                
                # 기존 컬렉션 삭제
                self.chroma_client.delete_collection(name=collection_name)
                logger.info(f"   기존 컬렉션 '{collection_name}' 삭제 완료")
                
                # 새 컬렉션 생성
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata=collection_metadata or {"description": "향상된 회사 전체 내규 벡터 검색 컬렉션"}
                )
                logger.info(f"   새 컬렉션 '{collection_name}' 생성 완료")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ 컬렉션 재생성 실패: {e}")
            
            # 방법 3: 새로운 이름으로 컬렉션 생성
            try:
                logger.info("🔄 방법 3: 새 이름으로 컬렉션 생성...")
                import time
                new_collection_name = f"company_regulations_{int(time.time())}"
                
                self.collection = self.chroma_client.create_collection(
                    name=new_collection_name,
                    metadata={"description": "향상된 회사 전체 내규 벡터 검색 컬렉션 (재생성)"}
                )
                logger.info(f"   새 컬렉션 '{new_collection_name}' 생성 완료")
                return True
                
            except Exception as e:
                logger.error(f"❌ 새 컬렉션 생성도 실패: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 컬렉션 정리 실패: {e}", exc_info=True)
            return False
    
    def build_index(self):
        """ChromaDB 인덱스 구축 (개선된 오류 처리)"""
        logger.info("ChromaDB 인덱스 구축 시작")
        try:
            if not hasattr(self, 'regulations_data') or not self.regulations_data:
                logger.warning("⚠️ 로드된 내규 데이터가 없습니다. 먼저 load_company_regulations_data()를 호출하세요.")
                return False

            existing_count = self.collection.count()
            if existing_count > 0:
                logger.info(f"🔍 기존 인덱스 발견: {existing_count}개 벡터")
                if existing_count == len(self.regulations_data):
                    logger.info("ℹ️ 인덱스가 이미 최신 상태입니다. 재구축을 건너뜀.")
                    return True
                else:
                    logger.info("🔄 내규 데이터가 변경되어 인덱스를 재구축합니다. 기존 데이터 삭제 중...")
                    if not self._clear_collection_safely():
                        logger.error("❌ 기존 데이터 삭제 실패")
                        return False
                    logger.info("✅ 기존 데이터 삭제 완료.")
            
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
            
            logger.info(f"⚙️ 텍스트 임베딩 및 ChromaDB 저장 중... (총 {len(texts)}개 청크)")
            
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
                    
                    logger.info(f"📦 배치 {i//batch_size + 1}/{total_batches} 처리 완료 ({len(batch_texts)}개 청크)")
                    
                except Exception as batch_error:
                    logger.error(f"❌ 배치 {i//batch_size + 1} 처리 실패: {batch_error}")
                    # 개별 항목 처리 시도
                    for j, (text, metadata, id_val) in enumerate(zip(batch_texts, batch_metadatas, batch_ids)):
                        try:
                            embedding = self.model.encode([text]).tolist()
                            self.collection.add(
                                documents=[text],
                                metadatas=[metadata],
                                embeddings=embedding,
                                ids=[id_val]
                            )
                        except Exception as item_error:
                            logger.warning(f"⚠️ 개별 항목 {i+j+1} 처리 실패: {item_error}")
                            continue
            
            final_count = self.collection.count()
            logger.info(f"✅ ChromaDB 인덱스 구축 완료: {final_count}개 벡터 저장됨.")
            return True
            
        except Exception as e:
            logger.error(f"❌ 인덱스 구축 실패: {e}", exc_info=True)
            return False
    
    def _generate_query_variants(self, query: str) -> List[str]:
        """쿼리 변형 생성으로 다양한 각도에서 검색"""
        variants = [query]  # 원본 쿼리
        
        # 의문사 제거한 버전
        question_words = ['무엇', '어떻게', '언제', '어디서', '왜', '누가', '얼마나', '몇']
        cleaned_query = query
        for word in question_words:
            cleaned_query = cleaned_query.replace(word, '').strip()
        if cleaned_query and cleaned_query != query:
            variants.append(cleaned_query)
        
        # 핵심 키워드 추출한 버전
        keywords = re.findall(r'\b[가-힣]{2,}\b', query)  # 한글 2글자 이상 키워드
        if len(keywords) >= 2:
            keyword_query = ' '.join(keywords[:3])  # 상위 3개 키워드
            variants.append(keyword_query)
        
        # 문제/상황 중심 변형
        if '문제' in query or '이슈' in query or '트러블' in query:
            variants.append(query.replace('문제', '해결').replace('이슈', '대응').replace('트러블', '처리'))
        
        # 절차/방법 중심 변형
        if '방법' in query or '절차' in query or '과정' in query:
            procedure_query = query.replace('방법', '절차 단계').replace('과정', '절차')
            variants.append(procedure_query)
        
        return list(set(variants))  # 중복 제거
    
    def search_with_enhanced_retrieval(self, query: str, top_k: int = 20, main_category_filter: str = None, min_relevance_score: float = 0.3) -> List[Dict[str, Any]]:
        """향상된 검색 - 다중 쿼리, 하이브리드 검색, 향상된 재랭킹 (확장된 정보량)"""
        logger.info(f"향상된 RAG 검색: '{query[:50]}...', 필터='{main_category_filter}', top_k={top_k}")
        try:
            if self.collection.count() == 0:
                logger.warning("⚠️ 인덱스가 비어있습니다.")
                return []
            
            # 1. 다중 쿼리 생성
            query_variants = self._generate_query_variants(query)
            logger.info(f"🔀 생성된 쿼리 변형: {len(query_variants)}개")
            
            # 2. 각 쿼리 변형으로 검색
            all_candidates = {}  # ID를 키로 하는 딕셔너리
            
            # 필터 조건
            where_condition = None
            if main_category_filter:
                where_condition = {"main_category": main_category_filter}
            
            # 더 많은 후보 검색 (top_k 20 → search_limit 60+)
            search_limit = max(top_k * 3, 60)  # top_k가 20이면 60개 검색
            
            for i, variant_query in enumerate(query_variants):
                logger.info(f"  🔍 검색 {i+1}/{len(query_variants)}: '{variant_query[:30]}...'")
                
                query_embedding = self.model.encode([variant_query]).tolist()
                
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=search_limit,
                    include=['documents', 'metadatas', 'distances'],
                    where=where_condition
                )
                
                # 결과 처리 및 통합
                if results['ids'] and len(results['ids'][0]) > 0:
                    for j in range(len(results['ids'][0])):
                        doc_id = results['ids'][0][j]
                        metadata = results['metadatas'][0][j]
                        distance = results['distances'][0][j]
                        
                        # 기본 유사도 점수 계산
                        similarity_score = max(0, 1 - (distance / 2))
                        
                        # 쿼리 변형별 가중치 (원본 쿼리에 더 높은 가중치)
                        query_weight = 1.0 if i == 0 else 0.8  # 원본 쿼리가 더 중요
                        weighted_score = similarity_score * query_weight
                        
                        # 기존 결과와 병합 (최고 점수 유지)
                        if doc_id not in all_candidates:
                            all_candidates[doc_id] = {
                                'metadata': metadata,
                                'best_similarity': weighted_score,
                                'query_matches': [i],
                                'distances': [distance]
                            }
                        else:
                            # 더 좋은 점수로 업데이트
                            if weighted_score > all_candidates[doc_id]['best_similarity']:
                                all_candidates[doc_id]['best_similarity'] = weighted_score
                            all_candidates[doc_id]['query_matches'].append(i)
                            all_candidates[doc_id]['distances'].append(distance)
            
            logger.info(f"  📊 통합된 후보: {len(all_candidates)}개")
            
            # 3. 향상된 재랭킹
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
                
                # 질문 매칭 점수
                question_match_count = len(query_words.intersection(question_words))
                question_match_ratio = question_match_count / max(len(query_words), 1)
                
                # 답변 매칭 점수
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
                
                # 다중 쿼리 매칭 보너스
                multi_query_bonus = len(set(query_matches)) * 0.1  # 여러 쿼리에서 발견된 경우 보너스
                
                # 키워드 매칭 보너스
                keyword_bonus = (question_match_ratio * 0.4) + (answer_match_ratio * 0.3)
                
                # 최종 점수 계산
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
            
            # 4. 점수 기반 정렬 및 다양성 확보
            enhanced_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 5. 다양성을 위한 중복 제거 (같은 질문의 다른 청크 타입 처리)
            seen_questions = set()
            diverse_results = []
            
            for candidate in enhanced_candidates:
                question = candidate['question']
                
                # 새로운 질문이거나, 기존 질문이지만 chunk_type이 qa_full인 경우 우선
                if question not in seen_questions:
                    diverse_results.append(candidate)
                    seen_questions.add(question)
                elif candidate['chunk_type'] == 'qa_full' and len(diverse_results) < top_k:
                    # 기존 결과에서 같은 질문의 다른 청크를 qa_full로 교체
                    for i, existing in enumerate(diverse_results):
                        if existing['question'] == question and existing['chunk_type'] != 'qa_full':
                            diverse_results[i] = candidate
                            break
                
                if len(diverse_results) >= top_k:
                    break
            
            # 최종 결과 정리
            final_results = diverse_results[:top_k]
            
            # 랭크 할당
            for i, result in enumerate(final_results):
                result['rank'] = i + 1
            
            logger.info(f"✅ 향상된 검색 완료: {len(final_results)}개 결과 (후보 {len(enhanced_candidates)}개에서 선별)")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 향상된 검색 실패: {e}", exc_info=True)
            return []
    
    def search(self, query: str, top_k: int = 20, main_category_filter: str = None, min_relevance_score: float = 0.3) -> List[Dict[str, Any]]:
        """기본 검색 메서드 - 향상된 검색으로 위임 (확장된 정보량)"""
        return self.search_with_enhanced_retrieval(query, top_k, main_category_filter, min_relevance_score)
    
    def combine_contexts_with_history(self, current_context: List[Dict[str, Any]], conversation_history: List[Dict[str, str]], max_total_context: int = 25) -> List[Dict[str, Any]]:
        """현재 검색 컨텍스트와 이전 대화의 컨텍스트를 자연스럽게 결합 - 대폭 확장된 컨텍스트"""
        logger.info(f"컨텍스트 결합: 현재 {len(current_context)}개, 이전 대화 {len(conversation_history)}개, 최대 {max_total_context}개")
        try:
            # 이전 대화에서 컨텍스트 추출
            previous_contexts = []
            if conversation_history:
                for msg in conversation_history:
                    if msg.get('role') == 'assistant' and msg.get('context'):
                        previous_contexts.extend(msg['context'])
            
            # 중복 제거 (질문 기준)
            seen_questions = set()
            unique_contexts = []
            
            # 1. 현재 검색 결과 우선 추가
            for ctx in current_context:
                question = ctx.get('question', '')
                if question and question not in seen_questions:
                    seen_questions.add(question)
                    ctx_copy = ctx.copy()
                    ctx_copy['source_type'] = 'current'
                    unique_contexts.append(ctx_copy)
            
            # 2. 이전 컨텍스트 중 관련성 높은 것만 추가 (임계값 낮춤)
            remaining_slots = max_total_context - len(unique_contexts)
            if remaining_slots > 0:
                # 이전 컨텍스트를 점수순으로 정렬
                previous_sorted = sorted(
                    previous_contexts, 
                    key=lambda x: x.get('score', 0), 
                    reverse=True
                )
                
                for ctx in previous_sorted:
                    if len(unique_contexts) >= max_total_context:
                        break
                    
                    question = ctx.get('question', '')
                    # 관련성 임계값을 낮춰서 더 많은 이전 컨텍스트 포함 (0.4 → 0.25)
                    if question and question not in seen_questions and ctx.get('score', 0) >= 0.25:
                        seen_questions.add(question)
                        ctx_copy = ctx.copy()
                        # 이전 컨텍스트의 점수를 약간 낮춰서 현재 컨텍스트보다 우선순위가 낮도록 조정
                        ctx_copy['score'] = ctx_copy.get('score', 0) * 0.85 
                        ctx_copy['source_type'] = 'previous'
                        unique_contexts.append(ctx_copy)
            
            # 최종 정렬
            unique_contexts.sort(key=lambda x: (x.get('source_type') == 'current', x.get('score', 0)), reverse=True)
            
            logger.info(f"✅ 컨텍스트 결합 완료: {len(unique_contexts)}개 (현재: {len(current_context)}, 이전 추가: {len(unique_contexts) - len(current_context)})")
            
            return unique_contexts
            
        except Exception as e:
            logger.error(f"❌ 컨텍스트 결합 실패: {e}", exc_info=True)
            return current_context
    
    def get_categories(self) -> Dict[str, Any]:
        """전체 카테고리 정보 반환"""
        if not hasattr(self, 'main_categories'):
            logger.warning("⚠️ 카테고리 데이터가 로드되지 않았습니다.")
            return {'main_categories': {}, 'total_main_categories': 0, 'total_regulations': 0}
        
        return {
            'main_categories': self.main_categories,
            'total_main_categories': len(self.main_categories),
            'total_regulations': len(self.regulations_data) if hasattr(self, 'regulations_data') else 0
        }
    
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
    
    def rebuild_index(self):
        """인덱스 완전 재구축 (개선된 오류 처리)"""
        logger.info("ChromaDB 인덱스 완전 재구축 시작")
        try:
            current_collection_name = getattr(self.collection, 'name', 'company_regulations')
            
            # 안전한 컬렉션 재생성
            try:
                logger.info(f"🗑️ 기존 컬렉션 '{current_collection_name}' 삭제 중...")
                self.chroma_client.delete_collection(name=current_collection_name)
                logger.info("✅ 기존 컬렉션 삭제 완료")
            except Exception as delete_error:
                logger.warning(f"⚠️ 기존 컬렉션 삭제 실패 (이미 없을 수 있음): {delete_error}")
            
            # 새 컬렉션 생성
            try:
                logger.info("➕ 새 컬렉션 생성 중...")
                self.collection = self.chroma_client.create_collection(
                    name="company_regulations",
                    metadata={"description": "향상된 회사 전체 내규 벡터 검색 컬렉션"}
                )
                logger.info("✅ 새 컬렉션 생성 완료")
            except Exception as create_error:
                logger.error(f"❌ 새 컬렉션 생성 실패: {create_error}")
                
                # 대안: 새로운 이름으로 컬렉션 생성
                import time
                backup_name = f"company_regulations_backup_{int(time.time())}"
                logger.info(f"🔄 대안: '{backup_name}' 이름으로 컬렉션 생성...")
                
                try:
                    self.collection = self.chroma_client.create_collection(
                        name=backup_name,
                        metadata={"description": "향상된 회사 전체 내규 벡터 검색 컬렉션 (백업)"}
                    )
                    logger.info(f"✅ 백업 컬렉션 '{backup_name}' 생성 완료")
                except Exception as backup_error:
                    logger.error(f"❌ 백업 컬렉션 생성도 실패: {backup_error}")
                    return False
            
            return self.build_index()
            
        except Exception as e:
            logger.error(f"❌ 인덱스 재구축 실패: {e}", exc_info=True)
            return False

class LLMClient:
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions"):
        """LLM 클라이언트 초기화"""
        self.api_url = api_url
        logger.info(f"LLM 클라이언트 초기화: API URL = {self.api_url}")
    
    def _create_enhanced_system_prompt(self, context: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> str:
        """대폭 확장된 정보량을 활용하는 자연스러운 대화 프롬프트 생성"""
        
        # 컨텍스트 품질 분석 (확장된 기준)
        high_quality = [c for c in context if c.get('score', 0) >= 0.7]
        medium_quality = [c for c in context if 0.5 <= c.get('score', 0) < 0.7]
        low_quality = [c for c in context if 0.25 <= c.get('score', 0) < 0.5]
        
        # 기본 역할 정의 (대폭 확장된 정보량 강조)
        system_prompt = f"""당신은 우리 회사의 내규를 정확히 아는 친근한 동료입니다. 직원들의 궁금한 점을 자연스럽지만 정확하게 답변해주세요.

🎯 핵심 원칙 (절대 준수):
• **대폭 확장된 정보 활용**: 제공된 {len(context)}개의 다양한 내규 정보를 종합적으로 활용하여 완전한 답변 제공
• **내규 기반 답변 필수**: 반드시 제공된 내규 정보만을 바탕으로 답변하세요
• **추측 금지**: 내규에 명시되지 않은 내용은 절대 추측하거나 만들어내지 마세요
• **다각도 검토**: 여러 관련 내규가 있을 때는 모두 검토하여 포괄적 답변 제공
• **정보 종합 능력**: 관련된 여러 규정을 자연스럽게 연결하여 설명
• **불확실시 명시**: 관련 내규가 없거나 불분명하면 "내규에서 확인이 어려워요"라고 명확히 말하세요
• **담당 부서 및 문의 안내**: 정확한 답변이 어려울 때는 경영지원팀 또는 감사팀 문의 안내하고 그외의 팀 또는 부서는 답변하지 말것
• **연차, 휴가 계산**: 연차일수 = min(기본연차일수 + ⌊ (근속연수(년) − 1) ÷ 2 ⌋, 25)

💬 자연스러운 대화 방식:
• 친근한 톤으로 설명하되, 내규 내용은 정확히 전달
• 여러 관련 정보가 있을 때는 "내규를 보니까요", "추가로 확인해보니까요", "관련해서 또 다른 규정도 있어요" 등 자연스럽게 연결
• 복잡한 내용은 "쉽게 말하면", "정리하면", "핵심만 말씀드리면" 등으로 풀어서 설명
• 이전 대화와 자연스럽게 연결하되, 내규 범위 내에서만
• 풍부한 정보를 체계적으로 정리하여 전달

📋 답변 방식 (확장된 정보량 기반):
• 내규에 명확한 답이 있으면 → 모든 관련 정보를 종합하여 완전하고 체계적인 설명
• 여러 규정이 관련되면 → 각각을 구분해서 정리하되, 연관성과 우선순위도 설명
• 부분적으로만 관련된 정보가 있으면 → 확실한 부분과 불확실한 부분을 명확히 구분
• 다양한 사례나 상황이 있으면 → 각 경우별로 나누어 설명
• 내규가 애매하거나 없으면 → "내규에서는 구체적으로 명시되어 있지 않아요" + 담당 부서 안내
• 절대 내규에 없는 내용을 추가하거나 임의로 해석하지 않기"""

        # 대폭 확장된 컨텍스트 정보 추가
        if context:
            system_prompt += f"\n\n📚 참고할 내규 정보 (대폭 확장된 종합 분석 - 총 {len(context)}개):\n"
            
            # 현재 검색 결과
            current_contexts = [c for c in context if c.get('source_type') == 'current']
            if current_contexts:
                system_prompt += f"\n[이번 질문 관련 내규 - 확장된 다각도 검색 결과: {len(current_contexts)}개]\n"
                for i, item in enumerate(current_contexts, 1):
                    score = item.get('score', 0)
                    chunk_type = item.get('chunk_type', 'qa_full')
                    query_matches = item.get('query_matches', 1)
                    
                    relevance = "매우 관련 높음" if score >= 0.8 else "관련 높음" if score >= 0.6 else "관련 있음" if score >= 0.4 else "부분 관련"
                    
                    system_prompt += f"{i}. [{item.get('main_category', 'N/A')} > {item.get('sub_category', 'N/A')}]\n"
                    system_prompt += f"   Q: {item.get('question', 'N/A')}\n"
                    system_prompt += f"   A: {item.get('answer', 'N/A')}\n"
                    system_prompt += f"   (관련도: {relevance}, 타입: {chunk_type}, 매칭: {query_matches}개 쿼리)\n\n"
                
                system_prompt += f"💡 총 {len(current_contexts)}개의 관련 내규를 발견했습니다. 이들을 종합적으로 검토하여 완전하고 체계적인 답변을 제공하세요.\n"
            
            # 이전 대화 관련 내규
            previous_contexts = [c for c in context if c.get('source_type') == 'previous']
            if previous_contexts:
                system_prompt += f"\n[이전 대화에서 언급된 관련 내규: {len(previous_contexts)}개]\n"
                for i, item in enumerate(previous_contexts, len(current_contexts) + 1):
                    score = item.get('score', 0)
                    relevance = "매우 관련 높음" if score >= 0.8 else "관련 높음" if score >= 0.6 else "관련 있음"
                    system_prompt += f"{i}. [{item.get('main_category', 'N/A')} > {item.get('sub_category', 'N/A')}]\n"
                    system_prompt += f"   Q: {item.get('question', 'N/A')}\n"
                    system_prompt += f"   A: {item.get('answer', 'N/A')}\n"
                    system_prompt += f"   (관련도: {relevance}, 이전 대화)\n\n"
            
            # 컨텍스트 품질에 따른 정확한 안내 (확장된 기준)
            if len(high_quality) == 0 and len(medium_quality) == 0:
                system_prompt += "\n⚠️ 중요: 이번 질문과 직접 관련된 내규를 찾기 어려웠습니다. 부분적으로 관련된 정보만 있으므로, 확실한 부분만 답변하고 불분명한 내용은 절대 추측하지 말며, 담당 부서 문의를 안내하세요.\n"
            elif len(high_quality) == 0:
                system_prompt += f"\n💡 주의: 관련성이 보통인 내규들이 {len(medium_quality)}개 있습니다. 이들을 종합적으로 검토하되, 확실한 부분만 답변하고, 불분명한 내용은 '내규에서 구체적으로 명시되어 있지 않다'고 명확히 표현하세요.\n"
            else:
                system_prompt += f"\n✅ 우수: 관련성이 높은 내규 {len(high_quality)}개를 포함하여 총 {len(context)}개의 풍부한 정보를 제공받았습니다. 이들을 종합적으로 활용하여 완전하고 정확한 답변을 제공하세요.\n"
                
            # 대폭 확장된 정보량에 대한 특별 안내
            if len(context) >= 20:
                system_prompt += f"\n🚀 특별 안내: 이번에는 {len(context)}개의 매우 풍부한 정보가 제공되었습니다. 이 모든 정보를 체계적으로 검토하여 가능한 한 완전하고 포괄적인 답변을 제공해주세요. 관련된 여러 규정이 있다면 우선순위와 연관성을 설명해주세요.\n"
        
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
    
    def generate_response_stream_with_history(self, query: str, context: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None):
        """대폭 확장된 정보량을 활용한 스트리밍 응답 생성"""
        logger.info(f"📞 대폭 확장된 정보량 기반 스트리밍 응답 생성: '{query[:50]}...' (컨텍스트: {len(context)}개)")
        
        try:
            # 대폭 확장된 시스템 프롬프트 생성
            system_content = self._create_enhanced_system_prompt(context, conversation_history)
            
            messages = [{"role": "system", "content": system_content}]
            
            # 이전 대화 추가 (최근 2-3개만)
            if conversation_history:
                recent_history_for_api = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history
                for msg in recent_history_for_api:
                    if msg.get('role') in ['user', 'assistant']:
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
            
            messages.append({"role": "user", "content": query})
            
            # LLM API 호출 - 대폭 확장된 정보량 처리 설정
            response = requests.post(
                self.api_url,
                json={
                    "model": "qwen3-30b-a3b-mlx",
                    "messages": messages,
                    "temperature": 0.1,  # 정확성 우선
                    "max_tokens": 3000,  # 2500 → 3000으로 확장 (더 많은 정보 처리)
                    "top_p": 0.85,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.05,
                    "stream": True
                },
                timeout=150,  # 120초 → 150초 (더 많은 정보 처리 시간)
                stream=True
            )
            
            if response.status_code == 200:
                accumulated_response = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data_str = line_text[6:]
                            
                            if data_str.strip() == '[DONE]':
                                logger.info(f"✅ 대폭 확장된 정보량 기반 스트리밍 응답 완료")
                                yield {"type": "done", "content": ""}
                                break
                                
                            try:
                                data = json.loads(data_str)
                                delta = data['choices'][0]['delta']
                                
                                if 'content' in delta:
                                    chunk_content = delta['content']
                                    accumulated_response += chunk_content
                                    yield {"type": "content", "content": chunk_content}
                                
                            except json.JSONDecodeError:
                                continue
                            except (KeyError, IndexError):
                                continue
                
                # 응답 완료 처리
                if accumulated_response and len(accumulated_response) >= 2500:
                    warning_msg = "\n\n💬 더 궁금한 점이 있으시면 언제든 다시 물어보세요!"
                    yield {"type": "warning", "content": warning_msg}
                yield {"type": "done", "content": ""}
            else:
                error_msg = "죄송해요, 지금 시스템에 문제가 있는 것 같아요. 잠시 후에 다시 시도해주세요."
                yield {"type": "error", "content": error_msg}
                
        except requests.exceptions.Timeout:
            error_msg = "대폭 확장된 정보를 처리하는 데 시간이 걸리고 있어요. 잠시 후에 다시 시도해주세요."
            yield {"type": "error", "content": error_msg}
        except Exception as e:
            logger.error(f"❌ 대폭 확장된 정보량 기반 스트리밍 응답 생성 실패: {e}", exc_info=True)
            error_msg = "죄송해요, 답변을 만드는 중에 문제가 생겼어요. 다시 시도해주세요."
            yield {"type": "error", "content": error_msg}

    def generate_response_with_history(self, query: str, context: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> str:
        """대폭 확장된 정보량을 활용한 비스트리밍 응답 생성"""
        logger.info(f"📞 대폭 확장된 정보량 기반 응답 생성: '{query[:50]}...' (컨텍스트: {len(context)}개)")

        try:
            # 대폭 확장된 시스템 프롬프트 생성
            system_content = self._create_enhanced_system_prompt(context, conversation_history)
            
            messages = [{"role": "system", "content": system_content}]
            
            # 이전 대화 추가
            if conversation_history:
                recent_history = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history
                for msg in recent_history:
                    if msg.get('role') in ['user', 'assistant']:
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
            
            messages.append({"role": "user", "content": query})

            # LLM API 호출 - 대폭 확장된 정보량 처리
            response = requests.post(
                self.api_url,
                json={
                    "model": "qwen3-30b-a3b-mlx",
                    "messages": messages,
                    "temperature": 0.1,  # 정확성 우선
                    "max_tokens": 3000,  # 2500 → 3000으로 확장
                    "top_p": 0.85,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.05,
                    "stream": True
                },
                timeout=150,  # 120초 → 150초
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
                                    generated_text += delta['content']
                                
                                if 'finish_reason' in data['choices'][0]:
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                            except (KeyError, IndexError):
                                continue
                
                if not generated_text.strip():
                    return "죄송해요, 답변을 생성하지 못했어요. 다시 시도해주세요."
                
                logger.info(f"✅ 대폭 확장된 정보량 기반 응답 생성 완료 (길이: {len(generated_text)}자)")
                return generated_text
            else:
                return "죄송해요, 지금 시스템에 문제가 있는 것 같아요. 잠시 후에 다시 시도해주세요."
                
        except requests.exceptions.Timeout:
            return "대폭 확장된 정보를 처리하는 데 시간이 걸리고 있어요. 잠시 후에 다시 시도해주세요."
        except Exception as e:
            logger.error(f"❌ 대폭 확장된 정보량 기반 응답 생성 실패: {e}", exc_info=True)
            return "죄송해요, 답변을 만드는 중에 문제가 생겼어요. 다시 시도해주세요."

# FastAPI 앱 설정
app = FastAPI(
    title="대폭 확장된 회사 내규 RAG 시스템",
    description="""
    **FastAPI 기반 대폭 확장된 정보량 회사 내규 RAG 시스템**
    
    ## 주요 특징
    - 🔍 다중 쿼리 검색 (원본 + 변형 쿼리로 다각도 검색)
    - 📝 향상된 청킹 전략 (QA + 질문 + 답변 + 문장 단위)
    - 🎯 하이브리드 재랭킹 (벡터 + 키워드 + 다중쿼리 매칭)
    - 📊 대폭 확장된 컨텍스트 윈도우 (최대 25개 → 훨씬 더 풍부한 정보)
    - 🌟 다양성 보장 알고리즘 (중복 제거 + 품질 유지)
    - 📉 낮은 관련성 임계값 (0.25, 부분 관련 정보도 포함)
    - 🎯 정확성 최우선 (Temperature: 0.1)
    - 📝 더 긴 응답 허용 (2500 토큰)
    
    ## 성능 지표 (대폭 개선)
    - **검색 범위**: top_k=20 (기존 8개 → 20개로 2.5배 확장)
    - **컨텍스트 윈도우**: 25개 (기존 10개 → 25개로 2.5배 확장)
    - **검색 한계**: 60개 후보 (기존 25개 → 60개로 2.4배 확장)
    - **최소 관련성**: 0.25 (더 많은 정보 수집)
    - **최대 토큰**: 2500 (완전한 답변 허용)
    - **Temperature**: 0.1 (정확성 우선)
    """,
    version="3.0.0",
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

# 전역 예외 처리기
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """요청 검증 오류 처리"""
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
                "common_issues": [
                    "query 필드가 누락되었거나 빈 문자열입니다.",
                    "conversation_history 형식이 올바르지 않습니다.",
                    "top_k 값이 1-50 범위를 벗어났습니다.",
                    "POST 요청시 Content-Type: application/json이 누락되었습니다.",
                    "테스트용 /test/simple_chat 엔드포인트 사용을 권장합니다."
                ]
            }
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """내부 서버 오류 처리"""
    logger.error(f"❌ 내부 서버 오류: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "내부 서버 오류가 발생했습니다.",
            "message": "서버 로그를 확인해주세요.",
            "request_id": id(request)
        }
    )

# 전역 객체들
rag_system: CompanyRegulationsRAGSystem = None
llm_client: LLMClient = None

def initialize_system():
    """시스템 초기화"""
    global rag_system, llm_client
    
    if rag_system is not None and llm_client is not None:
        logger.info("시스템이 이미 초기화되어 있습니다.")
        return

    logger.info("향상된 RAG 시스템 초기화 시작...")
    
    try:
        rag_system = CompanyRegulationsRAGSystem()
        llm_client = LLMClient()
        
        data_directory = "./data_json"
        
        if os.path.exists(data_directory):
            logger.info("회사 내규 데이터 로드 및 향상된 인덱스 구축...")
            if rag_system.load_company_regulations_data(data_directory):
                if rag_system.build_index():
                    stats = rag_system.get_stats()
                    logger.info(f"✅ 향상된 시스템 초기화 완료: {stats['total_documents']}개 청크")
                    logger.info(f"📊 청킹 통계: {stats.get('chunk_statistics', {})}")
                else:
                    logger.error("❌ 인덱스 구축 실패")
            else:
                logger.error("❌ 내규 데이터 로드 실패")
        else:
            logger.error(f"❌ 데이터 디렉토리 없음: {data_directory}. 데이터 로드를 건너뜁니다.")

    except Exception as e:
        logger.critical(f"🔥 향상된 시스템 초기화 실패: {e}", exc_info=True)
        rag_system = None
        llm_client = None

@app.get("/health", response_model=HealthResponse, summary="대폭 확장된 시스템 상태 확인", description="대폭 확장된 RAG 시스템의 상태와 통계를 확인합니다.")
async def health_check():
    """헬스 체크"""
    if rag_system is None or llm_client is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "initializing_or_error",
                "rag_ready": False,
                "regulations_count": 0,
                "main_categories_count": 0,
                "improvements": "대폭 확장된 정보량, 2.5배 검색 범위, 정확도 개선",
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
        improvements="대폭 확장된 정보량, 2.5배 검색 범위, 정확도 개선",
        enhanced_features=stats.get('enhanced_features', [])
    )

@app.get("/categories", summary="카테고리 정보 조회", description="전체 내규 카테고리 정보를 조회합니다.")
async def get_categories():
    """카테고리 정보 조회"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="대폭 확장된 RAG 시스템이 초기화되지 않았습니다.")
    
    categories_info = rag_system.get_categories()
    return {
        "categories": categories_info,
        "database": "ChromaDB (Massively Enhanced)",
        "system_type": "대폭 확장된 정보량 회사 내규 시스템"
    }

@app.post("/search", response_model=SearchResponse, summary="대폭 확장된 내규 검색", description="대폭 확장된 다중 쿼리 하이브리드 검색으로 관련 내규를 찾습니다.")
async def search_regulations(request: SearchRequest):
    """대폭 확장된 회사 내규 검색"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="대폭 확장된 RAG 시스템이 초기화되지 않았습니다.")
    
    # 대폭 확장된 검색 사용
    results = rag_system.search_with_enhanced_retrieval(
        request.query, 
        request.top_k,  # 기본값 20
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
            f"2.5배 확장된 검색 범위 (최대 {request.top_k}개)"
        ]
    )

@app.post("/chat", response_model=ChatResponse, summary="향상된 내규 상담", description="향상된 정보량을 기반으로 자연스러운 내규 상담을 제공합니다.")
async def chat_with_rag(request: ChatRequest):
    """향상된 정보량 기반 RAG 상담"""
    try:
        if not rag_system or not llm_client:
            raise HTTPException(status_code=503, detail="향상된 시스템이 초기화되지 않았습니다.")
        
        # conversation_history를 Dict 형식으로 변환
        conversation_history = []
        for msg in request.conversation_history:
            conversation_history.append({
                "role": msg.role,
                "content": msg.content,
                "context": getattr(msg, 'context', None)
            })
        
        # 향상된 검색 (대폭 확장된 결과)
        current_results = rag_system.search_with_enhanced_retrieval(
            request.query, 
            top_k=20,  # 8 → 20으로 확장
            main_category_filter=request.main_category_filter, 
            min_relevance_score=0.3  # 더 낮은 임계값
        )
        
        # 대폭 확장된 컨텍스트 결합
        combined_context = rag_system.combine_contexts_with_history(
            current_results, 
            conversation_history, 
            max_total_context=25  # 10 → 25로 확장
        )
        
        # 향상된 정보량 기반 응답 생성
        response = llm_client.generate_response_with_history(
            request.query, 
            combined_context, 
            conversation_history
        )
        
        # 품질 분석
        high_quality = len([c for c in combined_context if c.get('score', 0) >= 0.7])
        medium_quality = len([c for c in combined_context if 0.5 <= c.get('score', 0) < 0.7])
        low_quality = len([c for c in combined_context if 0.3 <= c.get('score', 0) < 0.5])
        
        context_results = [SearchResult(**ctx) for ctx in combined_context]
        
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
            search_type="향상된 다중쿼리 하이브리드 검색",
            response_type="대폭 확장된 정보량 기반 대화형",
            enhanced_features=[
                f"다중 쿼리 검색으로 {len(current_results)}개 정보 수집",
                f"총 {len(combined_context)}개 컨텍스트 활용 (기존 10개 → 25개)",
                "향상된 청킹 전략 적용",
                "다양성 보장 알고리즘 적용",
                "확장된 검색 범위 (top_k: 8→20)"
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 상담 처리 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"상담 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/chat_stream", summary="향상된 실시간 내규 상담", description="향상된 정보량을 기반으로 실시간 스트리밍 내규 상담을 제공합니다.")
async def chat_with_rag_stream(request: StreamChatRequest):
    """향상된 정보량 기반 스트리밍 상담"""
    try:
        if not rag_system or not llm_client:
            raise HTTPException(status_code=503, detail="향상된 시스템이 초기화되지 않았습니다.")
        
        # conversation_history 안전하게 처리
        conversation_history = request.conversation_history or []
        
        # 대폭 향상된 검색
        current_results = rag_system.search_with_enhanced_retrieval(
            request.query, 
            top_k=20,  # 8 → 20으로 확장
            main_category_filter=request.main_category_filter, 
            min_relevance_score=0.3
        )
        
        # 대폭 확장된 컨텍스트 결합
        combined_context = rag_system.combine_contexts_with_history(
            current_results, 
            conversation_history, 
            max_total_context=25  # 10 → 25로 확장
        )
        
        def generate():
            try:
                # 향상된 컨텍스트 정보 전송
                high_quality = len([c for c in combined_context if c.get('score', 0) >= 0.7])
                medium_quality = len([c for c in combined_context if 0.5 <= c.get('score', 0) < 0.7])
                low_quality = len([c for c in combined_context if 0.3 <= c.get('score', 0) < 0.5])
                
                context_data = {
                    "type": "context",
                    "query": request.query,
                    "context": combined_context,
                    "context_count": len(combined_context),
                    "context_quality": {
                        "high_relevance": high_quality,
                        "medium_relevance": medium_quality,
                        "low_relevance": low_quality
                    },
                    "search_type": "대폭 확장된 다중쿼리 하이브리드 검색",
                    "enhanced_features": [
                        f"다중 쿼리 검색으로 {len(current_results)}개 정보 수집 (기존 8개 → 20개)",
                        f"총 {len(combined_context)}개 컨텍스트 활용 (기존 10개 → 25개)",
                        "향상된 청킹 전략 적용",
                        "다양성 보장 알고리즘 적용",
                        "확장된 검색 범위로 더 풍부한 정보 제공"
                    ]
                }
                yield f"data: {json.dumps(context_data, ensure_ascii=False)}\n\n"
                
                # 향상된 정보량 기반 스트리밍 응답
                for chunk in llm_client.generate_response_stream_with_history(
                    request.query, 
                    combined_context, 
                    conversation_history
                ):
                    chunk_data = {
                        "type": chunk["type"],
                        "content": chunk["content"]
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                yield f"data: {json.dumps({'type': 'stream_end'}, ensure_ascii=False)}\n\n"
                
            except Exception as stream_error:
                logger.error(f"❌ 스트리밍 중 오류: {stream_error}", exc_info=True)
                error_data = {
                    "type": "error",
                    "content": f"스트리밍 중 오류가 발생했습니다: {str(stream_error)}"
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'stream_end'}, ensure_ascii=False)}\n\n"
        
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
            detail=f"스트리밍 상담 초기화 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/rebuild_index", summary="대폭 확장된 인덱스 재구축", description="내규 데이터를 다시 로드하고 대폭 확장된 인덱스를 재구축합니다.")
async def rebuild_index():
    """대폭 확장된 인덱스 재구축"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="대폭 확장된 RAG 시스템이 초기화되지 않았습니다.")
    
    if rag_system.load_company_regulations_data("./data_json"):
        if rag_system.rebuild_index():
            stats = rag_system.get_stats()
            return {
                "message": "대폭 확장된 인덱스가 성공적으로 재구축되었습니다",
                "regulations_count": stats['total_documents'],
                "chunk_statistics": stats.get('chunk_statistics', {}),
                "improvements": "대폭 확장된 정보량, 2.5배 검색 범위, 정확도 개선",
                "enhanced_features": stats.get('enhanced_features', []),
                "performance_boost": {
                    "search_top_k": "8 → 20 (2.5배 증가)",
                    "context_window": "10 → 25 (2.5배 증가)",
                    "search_candidates": "25 → 60 (2.4배 증가)"
                }
            }
        else:
            raise HTTPException(status_code=500, detail="대폭 확장된 인덱스 재구축 실패")
    else:
        raise HTTPException(status_code=500, detail="데이터 로드 실패")

@app.get("/stats", summary="대폭 확장된 시스템 통계", description="대폭 확장된 RAG 시스템의 상세 통계와 성능 지표를 확인합니다.")
async def get_stats():
    """대폭 확장된 통계 정보"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="대폭 확장된 RAG 시스템이 초기화되지 않았습니다.")
    
    stats = rag_system.get_stats()
    stats.update({
        "system_type": "대폭 확장된 정보량 기반 회사 내규 시스템",
        "improvements": [
            "다중 쿼리 검색 (원본 + 변형 쿼리)",
            "향상된 청킹 전략 (QA + 질문 + 답변 + 문장)",
            "하이브리드 재랭킹 (벡터 + 키워드 + 다중쿼리)",
            "대폭 확장된 컨텍스트 윈도우 (최대 25개)",
            "다양성 보장 알고리즘",
            "더 낮은 관련성 임계값 (0.25)",
            "정확성 최우선 (Temperature: 0.1)",
            "더 긴 응답 허용 (2500 토큰)",
            "2.5배 확장된 검색 범위 (8→20)"
        ],
        "performance_metrics": {
            "search_top_k": 20,
            "context_window": 25,
            "search_limit": 60,
            "min_relevance": 0.25,
            "max_tokens": 2500,
            "temperature": 0.1,
            "expansion_ratio": "2.5x"
        },
        "comparison_with_previous": {
            "search_top_k": "8 → 20 (2.5배 증가)",
            "context_window": "10 → 25 (2.5배 증가)",
            "search_candidates": "25 → 60 (2.4배 증가)",
            "min_relevance": "0.3 → 0.25 (더 포용적)",
            "expected_improvement": "답변 완성도 및 정확도 대폭 향상"
        }
    })
    return stats

# 사용자 정의 docs 설정 (선택사항)
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.0/bundles/redoc.standalone.js",
    )

# 간단한 테스트 엔드포인트
@app.post("/test/simple_chat", summary="간단한 채팅 테스트", description="간단한 형식으로 채팅 기능을 테스트합니다.")
async def simple_chat_test(request: SimpleTestRequest):
    """간단한 채팅 테스트 (복잡한 모델 없이)"""
    try:
        if not rag_system or not llm_client:
            raise HTTPException(status_code=503, detail="시스템이 초기화되지 않았습니다.")
        
        # 확장된 검색 수행
        results = rag_system.search_with_enhanced_retrieval(
            request.query, 
            top_k=15, 
            main_category_filter=request.category
        )
        
        # 간단한 응답 생성
        if results:
            response = llm_client.generate_response_with_history(request.query, results, [])
            return {
                "query": request.query,
                "response": response,
                "found_results": len(results),
                "status": "success",
                "info": "확장된 검색으로 더 풍부한 정보 제공 (top_k=15)"
            }
        else:
            return {
                "query": request.query,
                "response": "관련된 내규를 찾을 수 없습니다.",
                "found_results": 0,
                "status": "no_results"
            }
            
    except Exception as e:
        logger.error(f"❌ 간단한 채팅 테스트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"테스트 실패: {str(e)}")

@app.get("/test/stream_simple", summary="간단한 스트리밍 테스트", description="간단한 스트리밍 기능 테스트")
async def simple_stream_test(query: str = "휴가 신청 방법"):
    """간단한 스트리밍 테스트"""
    try:
        def generate():
            try:
                yield f"data: {json.dumps({'type': 'start', 'message': f'질문 처리 중: {query}'}, ensure_ascii=False)}\n\n"
                
                if not rag_system or not llm_client:
                    yield f"data: {json.dumps({'type': 'error', 'content': '시스템이 초기화되지 않았습니다.'}, ensure_ascii=False)}\n\n"
                    return
                
                # 확장된 검색
                results = rag_system.search_with_enhanced_retrieval(query, top_k=10)
                yield f"data: {json.dumps({'type': 'info', 'message': f'{len(results)}개 결과 발견 (확장된 검색)'}, ensure_ascii=False)}\n\n"
                
                # 스트리밍 응답
                for chunk in llm_client.generate_response_stream_with_history(query, results, []):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    
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

# 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    initialize_system()

if __name__ == '__main__':
    print("=" * 80)
    print("🏢 FastAPI 기반 대폭 확장된 정보량 회사 내규 RAG 서버")
    print("=" * 80)
    print("✨ 주요 개선사항:")
    print("   🔍 다중 쿼리 검색 (원본 + 변형 쿼리로 다각도 검색)")
    print("   📝 향상된 청킹 전략 (QA + 질문 + 답변 + 문장 단위)")
    print("   🎯 하이브리드 재랭킹 (벡터 + 키워드 + 다중쿼리 매칭)")
    print("   📊 대폭 확장된 컨텍스트 윈도우 (10개 → 25개, 2.5배)")
    print("   🌟 다양성 보장 알고리즘 (중복 제거 + 품질 유지)")
    print("   📉 더 낮은 관련성 임계값 (0.3 → 0.25, 더 많은 정보)")
    print("   🎯 정확성 최우선 (Temperature: 0.1)")
    print("   📝 더 긴 응답 허용 (2500 토큰)")
    print("   ⚡ 향상된 타임아웃 (120초)")
    print("   🛠️ FastAPI + Swagger UI 지원")
    print("   🚀 대폭 확장된 검색 범위 (8개 → 20개, 2.5배)")
    print("=" * 80)
    print("📊 성능 지표 비교:")
    print("   검색 결과 수     : 8개  → 20개  (2.5배 ⬆)")
    print("   컨텍스트 윈도우   : 10개 → 25개  (2.5배 ⬆)")
    print("   검색 후보 수     : 25개 → 60개  (2.4배 ⬆)")
    print("   관련성 임계값    : 0.3  → 0.25 (더 포용적)")
    print("=" * 80)
    print("🔧 FastAPI 엔드포인트:")
    print("   - GET  /health            : 대폭 확장된 시스템 상태 확인")
    print("   - POST /search            : 확장된 다중쿼리 하이브리드 검색")
    print("   - POST /chat              : 대폭 확장된 정보량 기반 상담")
    print("   - POST /chat_stream       : 대폭 확장된 실시간 상담 ⚡")
    print("   - POST /rebuild_index     : 향상된 인덱스 재구축")
    print("   - GET  /stats             : 확장된 성능 지표 포함 통계")
    print("   - POST /test/simple_chat  : 🧪 간단한 채팅 테스트 (top_k=15)")
    print("   - GET  /test/stream_simple: 🧪 간단한 스트리밍 테스트 (top_k=10)")
    print("   - GET  /docs              : 🎯 Swagger UI 문서 (API 테스트 가능) 🎯")
    print("   - GET  /redoc             : ReDoc 문서")
    print("=" * 80)
    print("📖 API 문서 및 테스트:")
    print("   http://localhost:5000/docs  ← 🎯 여기서 API 테스트하세요!")
    print("   http://localhost:5000/redoc ← 대안 문서")
    print("=" * 80)
    print("🧪 422 오류 해결 가이드:")
    print("   1. /test/simple_chat으로 먼저 테스트해보세요 (JSON body 사용)")
    print("   2. query 필드는 반드시 문자열이어야 합니다")
    print("   3. conversation_history는 빈 배열 []로 시작하세요")
    print("   4. /docs 페이지에서 'Try it out' 버튼 사용 권장")
    print("   5. POST 요청시 Content-Type: application/json 필수")
    print("=" * 80)
    print("📝 사용 예제 (확장된 검색):")
    print("   curl -X POST 'http://localhost:5000/test/simple_chat' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"query\": \"21년차 휴가\", \"category\": null}'")
    print("   또는")
    print("   curl -X POST 'http://localhost:5000/test/simple_chat' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"query\": \"휴가 신청 방법\", \"category\": \"인사\"}'")
    print("=" * 80)
    print("🎯 기대 효과:")
    print("   ✅ 2.5배 더 많은 관련 정보 수집")
    print("   ✅ 더 완전하고 정확한 답변 생성")
    print("   ✅ 복잡한 질문에 대한 포괄적 대답")
    print("   ✅ 놓치기 쉬운 관련 규정까지 포함")
    print("=" * 80)
    
    # 향상된 시스템 초기화
    initialize_system() 
    
    logger.info("🚀 대폭 확장된 FastAPI 서버 시작 중...")
    
    # 직접 FastAPI 앱 실행 (WSGIMiddleware 제거)
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level="info")