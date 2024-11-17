# Index

>[¶ 개요](#개요)<br>
>[¶ 문서 로드 기능](#문서-로드-기능)<br>
>[¶ 문서 분할 기능](#문서-분할-기능)<br>
>[¶ 문서 임베딩 기능](#문서-임베딩-기능)<br>
>[¶ Retriever 기능](#retriever)<br>
>[¶ RAG Chain 구성](#rag-chain-구성)<br>
>[¶ 기능 모듈화](#기능-모듈화)

## 개요

### 목표 : LLM과 RAG 기술을 활용해 사용자 질문에 답변하는 챗봇

### 구현 기능
1. LLM을 이용한 질문-답변 챗봇 제작
2. PDF 형식의 문서를 불러와 정보를 검색하는 RAG 구축

### 평가 환경
- **jupyter notebook**

### 사용 데이터
- 박찬준 외, 「초거대 언어모델 연구 동향」, 『정보학회지』, 제41권 제11호(통권 제414호), 한국정보과학회, 2023, 8-24
- 출처 [¶](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11610124)

### 개발환경

```
LLM_RAG/
│
├── README.md : 프로젝트 설명
├── requirements.txt : 패키지 목록
├── .gitignore : 버전관리 제외 목록
├── main.ipynb : 평가 환경
├── first_lot.py : 1차 작동 평가(단순 작동 평가)
├── second_lot.py : 2차 작동 평가(모듈화 작동 평가)
└── RAG_Module/
│   ├── RAG_Params.py : 사용할 파라미터를 `dataclass` 객체로 생성
│   ├── PDF_Loader.py
│   ├── VectorStore_Utils.py : 생성, 저장, 불러오기 기능
│   └── RAG_Chain.py : **Langchian** 생성 기능
├── documents/
│   └── 초거대 언어모델 연구 동향.pdf
└── testbed/
    ├── vector_store_index : 생성된 vectorstore 저장
    ├── load_documents.py : 문서 로드 기능 테스트
    ├── split_documents.py : 문서 분할 기능 테스트
    ├── vector_store.py : 분할된 문서 임베딩 테스트
    └── faiss_retriever.py : 주어진 쿼리(query)에 대한 검색 기능 테스트
```

[¶ Top](#index)
><br>

## 문서 로드 기능

### 사용가능 패키지

#### Fitz (PyMuPDF) Loader

- 이미지, 주석 등의 정보를 가져오는 데 매우 뛰어난 성능
- 페이지 단위
- PyMuPDF 라이브러리를 기반, 고해상도 이미지 처리 적합

```py
from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("example.pdf")
documents = loader.load()
```

#### PyPDFLoader

- PyPDF2 라이브러리를 사용하여 구현
- 경량, 빠르고 간단하게 텍스트 추출
- 구조화된 텍스트 추출
- 파일 크기가 큰 경우에도 효율적으로 처리

```py
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("example.pdf")
documents = loader.load()
```

#### UnstructuredPDFLoader

- Unstructured 라이브러리를 기반, 다양한 파일 형식을 처리
- 텍스트를 효율적으로 추출
- 비정형 데이터 처리에 강력한 성능을 발휘합니다.
- 특정 레이아웃이 없는 PDF 파일에서도 텍스트를 정확히 추출
- 문서의 구조에 덜 의존하는 방식

```py
from langchain.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader("example.pdf")
documents = loader.load()
```

#### PDFPlumberLoader

- 표와 같은 복잡한 구조의 데이터를 처리
- 텍스트, 이미지, 테이블, 필드 등 모두 추출 가능
- 문서의 메타데이터와 텍스트 간의 상호작용을 분석 

### 사용 패키지
- **PyPDFLoader**
  - 빠르고 간단하게 대용량 문서 처리 가능
  - 경량 패키지로 jupyter notebook 구동에도 적합할 것으로 판단

### 패키지 관리
- `pypdf 5.1.0` : pdf 파일 처리를 위해 설치
- `langchain_community-0.3.7` : **PyPDFLoader** 포함 라이브러리

### 코드 작성

```py
from langchain_community.document_loaders import PyPDFLoader

# 가져올 pdf 파일 주소
path = "documents/초거대 언어모델 연구 동향.pdf"

# 사용할 pdf loader 선택
loader = PyPDFLoader(path)

# pdf 파일 불러오기
docs = loader.load()

# 불러올 범위 설정
page = 0
start_point = 0
end_point = 500

# 결과 확인
print(f"페이지 수: {len(docs)}")
print(f"\n[페이지내용]\n{docs[page].page_content[start_point:end_point]}")
print(f"\n[metadata]\n{docs[page].metadata}\n")
```

### 출력 결과
- `page_content` : 인덱스별 출력 결과와 pdf파일 페이지별 내용과 일치
- `metadata` : 파일경로, 페이지 출력
  - `{'source': 'documents/초거대 언어모델 연구 동향 (1).pdf', 'page': 0}`

### 확인된 문제
- 페이지 제목이 최상단으로 이동함 

```
8 특집원고  초거대 언어모델 연구 동향
초거대 언어모델 연구 동향
업스테이지  박찬준*･이원성･김윤기･김지후･이활석...
```

- 각주 pdf 문서와 다른 위치에 출력됨
  - `ChatGPT1` 의 각주가 서론에도 붙어 출력됨

```
1. 서  론1)
ChatGPT1)와 같은 초거대 언어모델(Large Language 
Model, LLM) 의 등장으로
```

- 미주가 페이지 끝이 아닌 중간에 포함됨

```
이 모든 변화의 중심에는 ‘scaling law’라는 
* 정회원
1) https://openai.com/blog/chatgpt
학문적인 통찰이 있다
```

[¶ Top](#index)
><br>

## 문서 분할 기능

### 사용 가능 패키지
- **CharacterTextSplitter**
  - 기본적인 분할 방식
  - 구분자를 기준으로 청크 단위로 분할
- **RecursiveCharacterTextSplitter**
  - 단락-문장-단어 순서로 재귀적으로 분할
  - 여러번의 분할로 작은 덩어리 생성
  - 텍스트가 너무 크거나 복잡할 때 유용

### 사용 파라미터
1. `separator`
- 텍스트 분할을 위한 구분자로 사용되는 문자열
- 타입: 문자열
- `"\n\n"`: 두 개의 개행 문자 (기본값)
- `"\n"`: 한 개의 개행 문자
- `" "`: 공백
- `","`: 쉼표
- `"\t"`: 탭

2. `chunk_size`
- 분할 후 각 덩어리의 최대 크기
  - 기준 : 문자수
- 타입: 정수
3. `chunk_overlap`
- 덩어리 간 겹치는 부분의 크기
  - 문장이 끊겨서 의미를 알 수 없는 경우 보완
- 타입: 정수
4. `length_function`
- 타입: 함수
  - 기본 : len 함수
  - 필요한 함수를 작성하여 적용 가능
5. `is_separator_regex`
- 구분자의 정규 표현식인지 여부
- 타입: 불리언
  - `True`: 구분자를 정규 표현식으로 처리
  - `False`: 구분자를 문자열로 처리 

### 패키지 관리
- `langchain 0.3.7`

### 코드 작성

#### 파라미터 설정

```py
CHUNK_INDEX = 0
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
SEPERATOR = "\n"
```

#### CharacterTextSplitter

```py
from langchain.text_splitter import CharacterTextSplitter

# 문서 분할기 설정
splitter = CharacterTextSplitter(
    separator=SEPERATOR,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=True,
)

# 문서 분할
c_splits = splitter.split_documents(docs)

# 결과 확인
print(
    f"c_splits\n길이 : {len(c_splits)}\n 결과 확인 : {c_splits[CHUNK_INDEX].page_content}"
)
```

#### RecursiveCharacterTextSplitter

```py
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서 분할기 설정
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

# 문서 분할
r_splits = recursive_splitter.split_documents(docs)

# 결과 확인
print(
    f"r_splits\n길이 : {len(r_splits)}\n 결과 확인 : {r_splits[CHUNK_INDEX].page_content}"
)
```

#### Sample 테스트

```py
def sample_printer(splits_1, splits_2, n):
    for i in range(n):
        print(f"Sample 생성중...")
        print(f"Sample_1 {i} \n {splits_1[i].page_content}")
        print(f"Sample_2 {i} \n {splits_2[i].page_content}")
```

### 출력결과

#### 파라미터 1

```py
CHUNK_INDEX = 1
CHUNK_SIZE = 100
CHUNK_OVERLAP = 10
SEPERATOR = "\n\n"
```

- **CharacterTextSplitter**
  - 한 페이지 분량 출력
    - 본문에 `'\n\n'` 구분자로 나뉘는 부분이 없음
- **RecursiveCharacterTextSplitter**
  - 한 문단 분량 출력

#### 파라미터 2
- 구분자 `"\n"` 로 변경
- 결과가 같아짐

```py
CHUNK_INDEX = 1
CHUNK_SIZE = 100
CHUNK_OVERLAP = 10
SEPERATOR = "\n"
```

```py
# RecursiveCharacterTextSplitter
c_splits
길이 : 121

# CharacterTextSplitter
r_splits
길이 : 121
```

- **CharacterTextSplitter**
  - 한 문단 분량 출력
- **RecursiveCharacterTextSplitter**
  - 한 문단 분량 출력

#### 파라미터 3 
- 한 단락의 절반 분량

```py
CHUNK_INDEX = 0
CHUNK_SIZE = 500
CHUNK_OVERLAP = 10
SEPERATOR = "\n"
```

#### 파라미터 4
- 한 단락 분량
- 중첩 : 2~3 문단으로 맥락이 이어지게 설정
- 해당 파라미터로 진행

```py
CHUNK_INDEX = 0
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
SEPERATOR = "\n"
```

[¶ Top](#index)
><br>

## 문서 임베딩 기능

### 모델 정보 [¶](https://platform.openai.com/docs/guides/embeddings/)
- **MTEB bench** [¶](https://github.com/embeddings-benchmark/mteb)
  - 허깅페이스의 '대량 텍스트 임베딩 벤치마크 리더보드(MTEB)' 텍스트 검색 평가
- 사용모델 : `text-embedding-ada-002`

```
|------ 모델 ----------|-pages/$-|-MTEB-|
text-embedding-3-small	62,500	 62.3%
text-embedding-3-large	9,615	  64.6%
text-embedding-ada-002	12,500	 61.0%
```

### 패키지 관리
- `openai-1.54.4`
- `langchain-openai-0.2.8`
- `faiss-cpu-1.9.0`

### API Key 설정
- 환경변수에서 **API Key** 호출

```py
import os
import openai

openai.api_key = os.environ.get("________")
```

### 임베딩 모델 설정
- 모델 지정
- **API Key** 입력

```py
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=openai.api_key
)
```

### FAISS Vectorstore 생성

#### FAISS 객체 초기화 방식
- 더 많은 설정 가능
- 대용량 문서 처리가 요구되거나 성능을 최대한으로 끌어올려야 할 상황
- 인덱스나, 임베딩이 계산되어 있는 경우 빠르게 진행
- `docstore` :각 벡터와 연관된 문서 데이터를 메모리에 저장
- `index_to_docstore_id`: 특정 벡터가 어떤 문서와 연관되는지를 저장
- `Sample Text` : 임베딩 벡터의 길이를 확인하기 위한 샘플 텍스트
  - 실제 데이터의 임베딩 길이와 동일하게 설정하는 데 도움
  - 임의 문장으로 설정

```py
index = faiss.IndexFlatL2(len(embedding_model.embed_query("Sample Text")))
vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
```

#### `from_documents` 방식
- 간단하고 빠른 방식
- 일반적인 상황에서 많이 사용

```py
vectorstore = FAISS.from_documents(
  documents=splits, embedding=embedding_model
  )
```

#### UUID : Universally Unique Identifier
- 고유한 **ID**를 생성
- 128비트 숫자로, 이를 통해 거의 중복 없이 고유한 식별자를 생성
- 여러 문서를 처리하거나 분석하는 과정에서, 각 문서를 쉽게 추적하고 연결하려면 고유 식별자가 중요
- **metadata** 에 추가하는 방식 : 검색 시 참고
- **UUDI** 인덱스를 만들어 벡터 스토어에 등록하는 방식
  - 벡터 스토어의 고유 식별자로 사용 
  - 검색 시 **UUID** 기반으로 문서를 추적

```py
from uuid import uuid4

# 메타 데이터에 추가
for split in splits:
    split.metadata["uuid"] = str(uuid4())
vector_store.add_documents(documents=documents)

# UUID 인덱스 추가
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
```

#### Vector Store 저장
- `vector_store.save_local("testbed/vector_store_index/faiss_index")`
  - `index.faiss`
  - `index.pkl` : 객체를 바이너리 데이터 형태로 저장하는 피클 파일

## Retriever

### 패키지 관리
- `openai-1.54.4`
- `langchain-openai-0.2.8`
- `faiss-cpu-1.9.0`

### Vector Store 불러오기
- 생성 후 바로 사용하면 생략 가능
- 이미 만들어져 로컬에 저장된 **Vector Store**를 사용하는 방법
- `vector_store = FAISS.load_local("--path--", embedding_model)`

#### 확인된 문제
- `index.pkl` 파일을 불러오는 과정에서 발생하는 보안 경고로 코드 작동 중지
- **역직렬화 허용** 문제
  - **역직렬화** : `pkl` 피클파이에 저장된 데이터를 다시 **Python** 객체로 복원하는 과정
  - 악의적인 사용자가 파일을 수정하여 `pkl` 파일을 로드할 때 악성 코드가 실행되는 경우 존재
  - 신뢰할 수 있는 출처의 `pkl`만 사용하고 인터넷에서 다운로드한 경우 주의

#### 해결
- 직접 생성하거나 안전한 출처의 `pkl`의 경우 역직렬화를 허용하여 경고 무시
- `allow_dangerous_deserialization=True` : 해당 코드를 인자에 추가

### Retriever 생성
- `retriever = vector_store.as_retriever()`
  - `search_type` : 검색 기준
  - `search_kwargs` : `k` 찾을 문서 개수

#### 사용 파라미터
- `similarity` : 코사인 유사도 기반 검색
- `similarity_score_threshold` : `score_threshold` 이상만 검색
  - `search_kwargs` 설정
  - `score_threshold` 입력
- `mmr` : **maximum marginal search result** : 다양성을 고려한 검색
  - `search_kwargs` 설정
  - `fetch_k`: 후보 집합을 생성, 후보중에서 최종 `k`개 생성
  - `lambda_mult`: 유사도와 다양성 비중 조절
- `filter` : 메타 데이터를 기준으로 필터링, `search_kwargs`에 입력

### 코드 작성

```py
query = "RAG에 대해 이야기해주세요."

retriever = vector_store.as_retriever(search_type="similarity")
results = retriever.invoke(query)
for result in results:
    print(f"Source: {result.metadata['source']} | Page: {result.metadata['page']}")
    print(f"Content: {result.page_content.replace('\n', ' ')}\n")
```

### 결과
- **RAG** 에 관련된 문장을 적절하게 가져옴
- 사용된 **pdf** 파일의 레이아웃에 맞춰 `\n` 적용되어 있는 형태로 출력

```
Source: documents/초거대 언어모델 연구 동향.pdf | Page: 8
Content: 16 특집원고  초거대 언어모델 연구 동향
Retrieval Augmented Generation (RAG) [95, 96, 97, 
98]이라 한다.
Other Tools L a M D A  [ 4 4 ]는 대화 애플리케이션에 
특화된 LLM으로, 검색 모듈, 계산기 및 번역기 등의 
외부 도구 호출 기능을 가지고 있다. WebGPT [99] 는 
웹 브라우저와의 상호작용을 통해 검색 쿼리에 사실 
기반의 답변과 함께 출처 정보를 제공 한다......
```

[¶ Top](#index)
><br>

## RAG Chain 구성

### 사용 패키지
- `openai-1.54.4`
- `langchain-openai-0.2.8`
- `langchain-core-0.3.18`
- `faiss-cpu-1.9.0`

### LLM Model 설정

```py
llm_model = ChatOpenAI(model="gpt-4o-mini", api_key=openai.api_key)
```

### 프롬프트 작성
- `context`를 통해 전달 받은 정보를 참고
- `question` : 사용자의 질문

```py
from langchain_core.prompts import ChatPromptTemplate

contextual_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only the following context."),
        ("user", "Context: {context}\\n\\nQuestion: {question}"),
    ]
)
```

### 디버깅 함수
- `invoke` 메서드 사용한 결과를 전달
- 전달한 결과를 그대로 출력

```py
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        print("Debug Output:", output)
        return output
```

### `ContextToText` 함수
- 검색한 문서를 텍스트로 변환하여 전달

```py
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):
        context_text = [doc.page_content.replace('\n', ' ') for doc in inputs["context"]]
        return {"context": context_text, "question": inputs["question"]}
```

### Langchain 구성
- `|` : 파이프라인 사용, 파라미터를 순서대로 전달

```py
rag_chain_debug = (
    {"context": retriever, "question": DebugPassThrough()}
    | DebugPassThrough()
    | ContextToText()
    | contextual_prompt
    | llm_model
)
```

### 결과 확인

```py
response = rag_chain_debug.invoke("RAG에 대해 이야기해주세요.")
print("Final Response:")
print(response.content)
```

#### 확인된 문제
- **LLM** 모델이 답변을 제시하지 못하는 문제

#### 원인
- `ContextToText` 함수의 `context_text` 생성 코드
  - `'\n'.join()` 메서드 사용 시 문자 단위로 분할되어 `context` 로 활용되지 못함

#### 해결
`join` 메서드 제거 : 답변 생성
page_content 결과 내부 `'\n'` 제거 : 향상된 답변 생성

```
 \n 제거 전
Debug Output: RAG에 대해 이야기해주세요.

Final Response:
RAG는 Retrieval Augmented Generation의 약자로, 정보 검색 기능을 활용하여 생성 모델의 성능을 향상시키는 접근 방식입니다. 이 방법은 모델이 질문에 대한 답변을 생성할 때, 외부 데이터베이스나 검색 엔진에서 관련 정보를 검색한 후 이를 바탕으로 보다 정확하고 사실적인 응답을 생성할 수 있도록 돕습니다. RAG는 특히 대화형 AI나 질의응답 시스템에서 유용하게 사용됩니다.
```

```
\n 제거
Debug Output: RAG에 대해 이야기해주세요.

Final Response:
RAG, 즉 Retrieval Augmented Generation은 초거대 언어모델 연구의 한 동향으로, 정보 검색과 생성 과정을 결합하여 보다 정확하고 정보에 기반한 응답을 생성하는 방법입니다. 이 접근 방식은 기존의 언어 모델이 가지고 있는 한계점을 극복하고, 외부 데이터 소스에서 정보를 검색하여 그에 기반한 답변을 제공할 수 있도록 합니다. RAG는 특히 다양한 API와의 통합을 통해 계산, 번역, 검색 등의 기능을 활용하여 더 나은 성능을 발휘할 수 있습니다.
```

[¶ Top](#index)
><br>

## 기능 모듈화
> 작성한 코드들을 함수로 정리한 후 패키지로 구조화<br>

### 패키지 구조

```
RAG_Module/
├── RAG_Params.py
├── PDF_Loader.py
├── VectorStore_Utils.py
└── RAG_Chain.py
```

### RAG_Params
- 파라미터를 `dataclass` 객체로 정의
- 매개변수를 최소화하여 기능 사용

#### 확인된 문제
- 기본값이 없는 파라미터가 기본값이 있는 파라미터 뒤에 배치되면 오류 발생

#### 해결
- 기본값이 있는 파라미터 확인 및 명시 후 뒤에 배치

```py
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
```

### PDF_Loader
- `def PDFLoader(PARAMS, **kwargs):`

```
PDF 파일을 입력 받아 Document 객체 반환

PARAMS: RAGParams 객체
PDF_PATH : 사용 pdf 파일 경로
CHUNK_SIZE : 청크당 문자 수 
CHUNK_OVERLAP : 중첩시킬 문자 수
```

### VectorStore_Utils
- **Vector Store** 반환, 저장, 불러오기 기능

`def VectorStoreReturn(SPLITS, PARAMS, **kwargs):`

```
Document 객체를 임베딩하여 Vector Store 반환
SPLITS : Document 객체
PARAMS : RAGParams 객체
KEY : 환경변수에서 호출할 API Key 이름
EBD_MODEL : 임베딩 모델명
```

`def VectorStoreSave(SPLITS, PARAMS, **kwargs):`

```
Document 객체를 임베딩하여 Vector Store 저장
SPLITS : Document 객체
PARAMS : RAGParams 객체
KEY : 환경변수에서 호출할 API Key 이름
EBD_MODEL : 임베딩 모델명
SAVE_PATH : 저장할 경로
```

`def VectorStoreLoad(PARAMS, **kwargs):`

```
저장된 Vector Store 반환
SPLITS : Document 객체
PARAMS : RAGParams 객체
KEY : 환경변수에서 호출할 API Key 이름
EBD_MODEL : 임베딩 모델명
SAVE_PATH : Vector Store가 저장된 경로
IS_SAFE : 불러올 Vector Store이 안전한 파일인지 확인(불리언)
```

### RAG_Chain
- **RAG** 기법을 이용한 **LLM** 답변 구조 생성

`def RAGChainMake(VECTOR_STORE, PARAMS, **kwargs):`

```
VECTOR_STORE : Retriever가 검색할 벡터 스토어
KEY : 환경변수에서 호출할 API Key 이름
LLM_MODEL : 사용할 LLM 모델명
```

### 결과 확인

#### 파라미터 설정
- `params` 객체 생성 : 함수에 입력하면 필요한 매개변수만 입력됨

```py
from RAG_Module.RAG_Params import RAGParams

params = RAGParams(
    KEY= "_______",
    EBD_MODEL="text-embedding-ada-002",
    LLM_MODEL='gpt-3.5-turbo',
    PDF_PATH="documents/초거대 언어모델 연구 동향.pdf",
    SAVE_PATH = None,
    IS_SAFE=True,
    CHUNK_SIZE = 100,
    CHUNK_OVERLAP = 10
)
```

#### 답변 생성

```py
from RAG_Module.PDF_Loader import PDFLoader
from RAG_Module.VecotorStore_Utils import VectorStoreReturn
from RAG_Module.RAG_Chain import RAGChainMake

docs = PDFLoader(params)
vector_store = VectorStoreReturn(docs, params)
chatbot_mk1 = RAGChainMake(vector_store, params)

question = 'RAG에 대해서 설명해주세요'
response = chatbot_mk1.invoke(question)

print(response.content)
```

#### 답변 생성 결과
- 각 패키지가 문제 없이 작동
- 최소한의 청크 크기와 `gpt-3.5` 모델 사용으로 간단한 수준의 답변만 생성

```py
Load PDF.....
Load Complete....!
Split docs.....
Split Complete....!

Authenticate API KEY....
Authenticate Complete....!
Set up Embedding model....
Set up Complete....!
Initiate FAISS instance....
Return Vector Store....!

Debug Output: RAG에 대해서 설명해주세요
final response : RAG은 Retrieval Augmented Generation의 약자로, 초거대 언어모델 연구 동향 중 하나이다.
```

[¶ Top](#index)
><br>