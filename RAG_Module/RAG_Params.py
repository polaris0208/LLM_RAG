from dataclasses import dataclass

@dataclass
class RAGParams:
    KEY: str           # API Key 환경변수명
    EBD_MODEL: str     # 임베딩 모델명
    LLM_MODEL: str     # LLM 모델명, 기본값 없음
    PDF_PATH: str      # PDF 파일 경로, 기본값 없음
    SAVE_PATH: str = None  # 저장 경로 (옵션)
    IS_SAFE: bool = False  # 안전한 파일 로드 여부 (옵션)
    CHUNK_SIZE: int = 100  # 분할 크기 (기본값: 100)
    CHUNK_OVERLAP: int = 10  # 분할 중첩 크기 (기본값: 10)