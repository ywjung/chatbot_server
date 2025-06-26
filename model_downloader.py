#!/usr/bin/env python3
"""
GGUF 모델 다운로더
Qwen3-30B-A3B-Q4_K_M.gguf 모델을 안정적으로 다운로드하는 프로그램
"""

import os
import sys
import time
import json
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_download.log')
    ]
)
logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, models_dir: str = "./models"):
        """
        모델 다운로더 초기화
        
        Args:
            models_dir: 모델을 저장할 디렉토리
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # 세션 설정 (재시도 및 타임아웃)
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"모델 다운로더 초기화 완료. 저장 디렉토리: {self.models_dir}")
    
    def get_model_info(self, repo_id: str) -> Dict[str, Any]:
        """HuggingFace 저장소의 모델 정보 조회"""
        try:
            api = HfApi()
            repo_info = api.repo_info(repo_id)
            
            files = []
            for sibling in repo_info.siblings:
                if sibling.rfilename.endswith('.gguf'):
                    files.append({
                        'filename': sibling.rfilename,
                        'size': getattr(sibling, 'size', 0),
                        'size_mb': round(getattr(sibling, 'size', 0) / (1024 * 1024), 1) if hasattr(sibling, 'size') else 'Unknown'
                    })
            
            return {
                'repo_id': repo_id,
                'total_files': len(files),
                'gguf_files': files
            }
        except Exception as e:
            logger.error(f"모델 정보 조회 실패: {e}")
            return {}
    
    def download_with_progress(self, url: str, dest_path: Path, resume: bool = True) -> bool:
        """진행률 표시와 함께 파일 다운로드"""
        try:
            headers = {}
            initial_pos = 0
            
            # 기존 파일이 있고 resume=True인 경우 이어받기
            if resume and dest_path.exists():
                initial_pos = dest_path.stat().st_size
                headers['Range'] = f'bytes={initial_pos}-'
                logger.info(f"기존 파일 발견. {initial_pos} 바이트부터 이어받기 시작")
            
            response = self.session.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # 전체 파일 크기 계산
            content_length = response.headers.get('content-length')
            if content_length:
                total_size = int(content_length) + initial_pos
            else:
                total_size = None
                logger.warning("파일 크기를 알 수 없습니다.")
            
            # 파일 쓰기
            mode = 'ab' if resume and initial_pos > 0 else 'wb'
            
            with open(dest_path, mode) as f:
                with tqdm(
                    initial=initial_pos,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest_path.name
                ) as pbar:
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"✅ 다운로드 완료: {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 다운로드 실패: {e}")
            return False
    
    def download_with_hf_hub(self, repo_id: str, filename: str, max_retries: int = 3) -> Optional[Path]:
        """HuggingFace Hub을 사용한 다운로드"""
        local_dir = self.models_dir / repo_id.replace("/", "--")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 HuggingFace Hub 다운로드 시도 {attempt + 1}/{max_retries}")
                logger.info(f"   저장소: {repo_id}")
                logger.info(f"   파일: {filename}")
                logger.info(f"   저장 위치: {local_dir}")
                
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    resume_download=True,
                    force_download=False,
                    local_files_only=False
                )
                
                if Path(downloaded_file).exists():
                    logger.info(f"✅ HuggingFace Hub 다운로드 성공: {downloaded_file}")
                    return Path(downloaded_file)
                
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace Hub 다운로드 실패 (시도 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logger.info(f"⏳ {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
        
        return None
    
    def download_with_snapshot(self, repo_id: str, max_retries: int = 2) -> Optional[Path]:
        """전체 저장소 다운로드"""
        local_dir = self.models_dir / repo_id.replace("/", "--")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 전체 저장소 다운로드 시도 {attempt + 1}/{max_retries}")
                
                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    resume_download=True
                )
                
                logger.info(f"✅ 전체 저장소 다운로드 성공: {downloaded_path}")
                return Path(downloaded_path)
                
            except Exception as e:
                logger.warning(f"⚠️ 전체 저장소 다운로드 실패 (시도 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 15
                    logger.info(f"⏳ {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
        
        return None
    
    def verify_file(self, file_path: Path, expected_size: Optional[int] = None) -> bool:
        """다운로드된 파일 검증"""
        if not file_path.exists():
            logger.error(f"❌ 파일이 존재하지 않음: {file_path}")
            return False
        
        actual_size = file_path.stat().st_size
        logger.info(f"📊 파일 크기: {actual_size:,} bytes ({actual_size / (1024**3):.2f} GB)")
        
        if expected_size and actual_size != expected_size:
            logger.warning(f"⚠️ 파일 크기 불일치. 예상: {expected_size:,}, 실제: {actual_size:,}")
            return False
        
        # 파일이 올바른 GGUF 형식인지 확인
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if header.startswith(b'GGUF'):
                    logger.info("✅ 올바른 GGUF 파일 형식 확인")
                    return True
                else:
                    logger.error("❌ 올바르지 않은 GGUF 파일 형식")
                    return False
        except Exception as e:
            logger.error(f"❌ 파일 검증 중 오류: {e}")
            return False
    
    def download_model(self, repo_id: str, filename: str, force_redownload: bool = False) -> bool:
        """
        모델 다운로드 메인 함수
        
        Args:
            repo_id: HuggingFace 저장소 ID
            filename: 다운로드할 파일명
            force_redownload: 기존 파일이 있어도 다시 다운로드할지 여부
        
        Returns:
            bool: 다운로드 성공 여부
        """
        print("=" * 80)
        print(f"🤖 GGUF 모델 다운로더")
        print("=" * 80)
        print(f"📦 저장소: {repo_id}")
        print(f"📄 파일: {filename}")
        print(f"💾 저장 위치: {self.models_dir}")
        print("=" * 80)
        
        # 최종 파일 경로
        final_dir = self.models_dir / repo_id.replace("/", "--")
        final_path = final_dir / filename
        
        # 기존 파일 확인
        if final_path.exists() and not force_redownload:
            logger.info(f"✅ 파일이 이미 존재합니다: {final_path}")
            if self.verify_file(final_path):
                print(f"✅ 다운로드 완료! 파일 위치: {final_path}")
                return True
            else:
                logger.warning("⚠️ 기존 파일이 손상되었습니다. 다시 다운로드합니다.")
        
        # 모델 정보 조회
        logger.info("🔍 모델 정보 조회 중...")
        model_info = self.get_model_info(repo_id)
        if model_info:
            print(f"📊 저장소 정보:")
            print(f"   - 전체 GGUF 파일: {model_info['total_files']}개")
            for file_info in model_info['gguf_files']:
                marker = " ⭐" if file_info['filename'] == filename else ""
                print(f"   - {file_info['filename']}: {file_info['size_mb']} MB{marker}")
            print()
        
        # 다운로드 시도 1: HuggingFace Hub (권장)
        logger.info("🚀 다운로드 시작...")
        downloaded_path = self.download_with_hf_hub(repo_id, filename)
        
        if downloaded_path and self.verify_file(downloaded_path):
            print(f"✅ 다운로드 성공! 파일 위치: {downloaded_path}")
            return True
        
        # 다운로드 시도 2: 전체 저장소 다운로드
        logger.info("🔄 대안 방법으로 전체 저장소 다운로드 시도...")
        downloaded_path = self.download_with_snapshot(repo_id)
        
        if downloaded_path:
            target_file = downloaded_path / filename
            if target_file.exists() and self.verify_file(target_file):
                print(f"✅ 다운로드 성공! 파일 위치: {target_file}")
                return True
        
        # 모든 시도 실패
        logger.error("❌ 모든 다운로드 방법이 실패했습니다.")
        print("\n" + "=" * 80)
        print("❌ 다운로드 실패")
        print("=" * 80)
        print("💡 문제 해결 방법:")
        print("1. 인터넷 연결 확인")
        print("2. VPN 사용 시 비활성화")
        print("3. DNS 설정 확인 (8.8.8.8, 8.8.4.4)")
        print("4. HuggingFace CLI로 수동 다운로드:")
        print(f"   huggingface-cli download {repo_id} {filename} --local-dir {final_dir}")
        print("=" * 80)
        return False

def main():
    parser = argparse.ArgumentParser(description="GGUF 모델 다운로더")
    parser.add_argument(
        "--repo-id", 
        default="lmstudio-community/Qwen3-30B-A3B-GGUF",
        help="HuggingFace 저장소 ID"
    )
    parser.add_argument(
        "--filename", 
        default="Qwen3-30B-A3B-Q4_K_M.gguf",
        help="다운로드할 파일명"
    )
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="모델을 저장할 디렉토리"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 파일이 있어도 다시 다운로드"
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="사용 가능한 GGUF 파일 목록만 표시"
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir)
    
    # 파일 목록만 표시
    if args.list_files:
        print("🔍 사용 가능한 GGUF 파일 조회 중...")
        model_info = downloader.get_model_info(args.repo_id)
        if model_info:
            print(f"\n📦 {args.repo_id}")
            print("=" * 60)
            for file_info in model_info['gguf_files']:
                print(f"📄 {file_info['filename']}")
                print(f"   크기: {file_info['size_mb']} MB")
                print()
        return
    
    # 모델 다운로드
    success = downloader.download_model(args.repo_id, args.filename, args.force)
    
    if success:
        print("\n🎉 다운로드가 완료되었습니다!")
        print("이제 RAG 서버를 실행할 수 있습니다.")
        sys.exit(0)
    else:
        print("\n❌ 다운로드에 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()