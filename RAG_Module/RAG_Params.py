from dataclasses import dataclass


@dataclass
class RAGParams:
    KEY: str  # API Key 환경변수명
    EBD_MODEL: str  # 임베딩 모델명
    LLM_MODEL: str  # LLM 모델명, 기본값 없음
    PDF_PATH: str  # PDF 파일 경로, 기본값 없음
    SAVE_PATH: str = None  # 저장 경로 (옵션)
    IS_SAFE: bool = False  # 안전한 파일 로드 여부 (옵션)
    CHUNK_SIZE: int = 100  # 분할 크기 (기본값: 100)
    CHUNK_OVERLAP: int = 10  # 분할 중첩 크기 (기본값: 10)


@dataclass
class PromptParams:
    KEY: str  # API Key 환경변수명
    LLM_MODEL: str  # LLM 모델명
    PROMPT_PATH: str  # 프롬프트 파일 경로
    PROMPT_NAME: str  # 프롬프트 파일 이름
    PROMPT_EXTENSION: str  # 프롬프트 파일 확장자
    RESULT_PATH: str  # 결과 파일 경로
    RESULT_EXTENSION: str  # 결과 파일 확장자


@dataclass
class TemplateParams:
    PERSONA: str    # LLM이 수행할 역할 지정
    LANG: str   # 답변 생성 언어
    TONE: str   # 답변의 어조 설정
    PERPOSE: str    # 목적 명시
    HOW_WRITE: str  # 답변 방식 예) 개조식
    CONDITION: str  # 추가할 조건
    REFERENCE: str  # 참조
