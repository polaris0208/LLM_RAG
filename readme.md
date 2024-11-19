# Index

>[¶ 개요](#개요)<br>
>[¶ 문서 로드 기능](#문서-로드-기능)<br>
>[¶ 문서 분할 기능](#문서-분할-기능)<br>
>[¶ 문서 임베딩 기능](#문서-임베딩-기능)<br>
>[¶ Retriever 기능](#retriever)<br>
>[¶ RAG Chain 구성](#rag-chain-구성)<br>
>[¶ 기능 모듈화](#기능-모듈화)<br>
>[¶ 프롬프트 엔지니어링](#프롬프트-엔지니어링)<br>
>[¶ 최종 작동 평가](#최종-작동-평가)

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
├── documents/
│   └── 초거대 언어모델 연구 동향.pdf
│
├── Prompts/ : 프롬프트 저장 경로
├── Results/ : 프롬프트 실행 결과 저장 경로
├── main.ipynb : 제출용 jupyter notebook 파일
│
│── test_done/ : 테스트가 끝난 파일 보관
│   ├── first_lot.py : 1차 작동 평가 - 기능 작동 확인
│   ├── second_lot.py : 2차 작동 평가 - 모듈화 작동 확인
│   ├── third_lot.py : 3차 작동 평가 - 프롬프트 입력 및 대화형 응답 작동 확인
|   └── auto_test.py : 프롬프트 입력 및 결과 저장 자동화 작동 확인
│
├── RAG_Module/ : RAG 작동 패키지
│   ├── setup.py : RAG_Module 패키지 설정 파일
│   ├── PDF_Loader.py
│   ├── RAG_Params.py : 사용할 파라미터를 dataclass 객체로 생성
│   ├── VectorStore_Utils.py : 생성, 저장, 불러오기 기능
│   ├── Prompt_Engineering.py : 프롬프트 생성에 필요한 기능
│   ├── RAG_Chain.py : 개선된 Langchian 생성 기능
|   └── RAG_Chain_exhausted.py : 개선 전 코드
│
└── testbed/ : 코드 작동 테스트
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

### 패키지 관리
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

## 프롬프트 엔지니어링

### 파라미터 설정
- 프롬프트 생성에 필요한 파라미터 관리
- `class PromptParams` : 프롬프트 생성, 저장, 불러오기에 필요한 파라미터
- `class TemplateParams` : 프롬프트를 지정한 양식에 맞춰 생성하기 위한 파라미터

```py
from dataclasses import dataclass

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
```

### Prompt_Engineering 모듈
- `def PromptSave(PROMPT, PARAMS, PROMPT_NAME=None, **kwargs):`
  - 주어진 프롬프트를 파일에 저장하는 함수
  - 파일명은 개별 선언할 수 있게하여, 여러파일을 처리할 수 있도록 작성

```
PROMPT_PATH: str  # 프롬프트 파일 경로
PROMPT_NAME: str  # 프롬프트 파일 이름
PROMPT_EXTENSION: str  # 프롬프트 파일 확장자

PROMPT_NAME이 지정되지 않을 때 PARAMS.PROMPT_NAME을 사용.
```

- `def PromptLoad(PARAMS, PROMPT_NAME=None, **kwargs):`
  - 저장된 프롬프트를 불러오는 함수
  - 파일명은 개별 선언할 수 있게하여, 여러파일을 처리할 수 있도록 작성

```
PROMPT_PATH: str  # 프롬프트 파일 경로
PROMPT_NAME: str  # 프롬프트 파일 이름
PROMPT_EXTENSION: str  # 프롬프트 파일 확장자

PROMPT_NAME이 지정되지 않을 때 PARAMS.PROMPT_NAME을 사용.
```

- `def PromptResult(RESPONSE, PARAMS, PROMPT_NAME=None, **kwargs):`
  - 대화 결과를 파일에 저장하는 함수
  - `RESPONSE`가 문자열일 경우 그대로 저장
  - 리스트나 딕셔너리일 경우 `JSON` 형식으로 저장

```
PROMPT_PATH: str  # 프롬프트 파일 경로
PROMPT_NAME: str  # 프롬프트 파일 이름
PROMPT_EXTENSION: str  # 프롬프트 파일 확장자
RESULT_PATH: str  # 결과 파일 경로
RESULT_EXTENSION: str  # 결과 파일 확장자

PROMPT_NAME이 지정되지 않을 때 PARAMS.PROMPT_NAME을 사용.
```

- `def LLMSupport(QUESTION, PARAMS, SUPPORT_PROMPT="answser question", **kwargs):`
  - **shot** 기법 - **shot** 생성을 위한 함수
  - **LLM** 에게 예시를 생성하게 하여 프롬프트에 적용
  - 성능이 높은 모델로 예시 생성 - 성능이 낮은 모델에게 예시로 적용

```
QUESTION : shot 생성을 위한 질문
SUPPORT_PROMPT : shot 생성을 위한 프롬프트(선택)
KEY: str  # API Key 환경변수명
LLM_MODEL: str  # LLM 모델명
```

- `def PromptTemplate(PARAMS, **kwargs):`
  - `class TemplateParams` 객체를 파라미터로 입력
  - 프롬프트 작성에 필요한 요소들 정리
  - 필요 없는 부분은 공백('')으로 작성
  - 영어 사용, 명령어 사용, 리스트, 마크다운 작성 방식으로 성능을 향상 가능

```
PERSONA : LLM이 수행할 역할 지정
LANG : 답변 생성 언어
TONE : 답변의 어조 설정
PURPOSE : 목적 명시
HOW_WRITE : 답변 방식 예) 개조식
CONDITION : 추가할 조건
REFERENCE : 참조
```

### 프롬프트 템플릿 
- 역할, 목적, 답변 방식을 지정하여 전달
- 답변 생성 시 지켜야 할 조건 명시
- 답변에 참고할 내용 명시

```py
question = "RAG에 대해서 설명해주세요"
shot = LLMSupport(question, prompt_setting)


template_setting = TemplateParams(
    PERSONA="specialist of large language model",
    LANG="korean",
    TONE="professional",
    PERPOSE="study large language model",
    HOW_WRITE="itemization",
    CONDITION="""

    answer is about large language model
    prioritize the context of the question
    specify if there are any new information
    If you can identify the original source of the document you referenced, write in APA format

    """,
    REFERENCE=f"{shot} \n\n and context given in the question",
)

prompt = PromptTemplate(template_setting)
chatbot_mk3 = RAGChainMake(vector_store, rag_setting, prompt)
RAG_Coversation(chatbot_mk3, prompt_setting)
```

#### 결과
- 개조식으로 답변
- 전문 용어 사용과 구조화된 답변
- 찾은 여러가지의 지식 중 조건에 해당하는 지식을 구분하여 답변

```
질문을 입력하세요 :  RAG에 대해서 설명해주세요.

답변:
**RAG (Retrieval Augmented Generation)**에 대해 설명드리겠습니다:

- RAG는 위험 평가 격자를 의미하는 약자가 아닌, **Retrieval Augmented Generation**의 약자입니다.
- RAG는 주로 자연어 처리 분야에서 활용되며, 최근 초거대 언어모델 연구 동향에서 주목을 받고 있습니다.
- RAG는 정보 검색 (Retrieval)과 생성 (Generation)을 결합한 모델로, 정보를 검색하여 새로운 내용을 생성하는 기술을 지칭합니다.
- 이 모델은 이전에 생성된 텍스트나 문맥을 활용하여 보다 의미 있는 내용을 생성하고자 하는 데 사용됩니다.
- RAG는 자연어 이해, 생성, 정보 검색 등 다양한 작업에 활용되며, 다양한 응용 분야에서 유망한 기술로 평가되고 있습니다. 

이렇게, RAG는 자연어 처리 분야에서 중요한 기술 중 하나로 발전하고 있으며, 정보 검색과 생성을 결합한 혁신적인 모델로 주목받고 있습니다.
```

### 확인 된 문제
- 프롬프트 결과 저장 중 오류 발생

#### 원인
- 질문과 답변이 두 개 이상 이어지면 결과가 리스트 또는 딕셔너리의 형태로 저장됨
- 리스트 또는 딕셔너리를 `txt` 형태로 저장하기 위해서는 별도의 변환 과정 필요

#### 해결
- 프롬프트 결과 저장 함수를 수정
- 리스트나 딕셔너리 형식은 `json` 확장자로 변경하여 저장

```py
    # RESPONSE가 문자열일 경우
    if isinstance(RESPONSE, str):
        # 문자열을 그대로 저장
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(RESPONSE)

    # RESPONSE가 리스트나 딕셔너리일 경우
    elif isinstance(RESPONSE, (list, dict)):
        # 리스트나 딕셔너리를 JSON 형식으로 저장
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(RESPONSE, file, ensure_ascii=False, indent=4)

    else:
        print("지원하지 않는 RESPONSE 형식입니다.")
        return

    print(f"결과가 {file_path}에 {PARAMS.RESULT_EXTENSION} 형식으로 저장되었습니다.\n")
```

### 프롬프트 고도화
- 프롬프트 수정에 따른 답변 변화 확인
- `"question": "RAG에 대해서 설명해주세요"`

#### Prompt 1
- **RAG** 결과를 활용하도록 유도

```
Answer the question using only the following context.
```

```
"answer": "RAG은 Retrieval Augmented Generation의 약자로, 정보 검색을 통해 생성 모델을 보강하는 방법을 가리킨다."
```

#### Prompt 2
- 역할 지정
- **RAG** 결과를 활용하도록 구체적으로 유도

```
you are specialist of large language model
answer question
refer to context qiven in question
```

```
"answer": "RAG은 Retrieval Augmented Generation의 약자로, 정보 검색을 통해 생성 모델을 보강하는 방법론을 가리키는 용어입니다. 이 방법론은 정보 검색 기술을 활용하여 대규모 언어모델을 더욱 효과적으로 학습하고 활용할 수 있도록 설계되었습니다. RAG은 최근 자연어 처리 분야에서 주목을 받고 있으며, 관련 연구들이 활발히 진행되고 있습니다."
```

#### Prompt 3
- 역할, 답변 방식, 조건 구체적으로 명시
- **LLM** 생성한 답변 참고
- 결과
  - 역할에 맞게 전문용어를 사용하여 구조적으로 설명
  - **LLM** 의 답변 형태를 참고하여 답변 생성
  - 검색된 두 종류의 **RAG** 정보 중 **LLM** 과 관련된 정보만 답변에 포함

```
persona : specialist of large language model
language : korean
tone : professional
purpose : study large language model
how to write : itemization
condition : 

answer is about large language model
prioritize the context of the question
specify if there are any new information
If you can identify the original source of the document you referenced, write in APA format

    
reference : RAG는 Risk Assessment Grid의 약자로, 위험 평가 격자를 의미합니다. 이는 프로젝트나 활동을 평가하고 위험을 식별하기 위해 사용되는 도구입니다. RAG는 주로 세 가지 색상으로 표시되며, 각 색상은 다음을 나타냅니다:

- 빨강(Red): 심각한 위험이 있음을 나타냅니다. 이는 프로젝트나 활동이 큰 위험에 직면하고 있음을 의미합니다.
- 주황(Amber): 중간 수준의 위험이 있음을 나타냅니다. 이는 일부 위험이 존재하며 주의가 필요함을 의미합니다.
- 녹색(Green): 낮은 수준의 위험이 있음을 나타냅니다. 이는 프로젝트나 활동이 안정적이며 큰 위험 없이 진행될 수 있음을 의미합니다.

RAG를 사용하여 위험을 시각적으로 파악하고 관리자나 이해관계자에게 위험 상황을 전달하는 데 도움이 됩니다. 이를 통해 조직이 적절한 대응책을 마련하고 위험을 최소화할 수 있습니다. 

 and context given in the question
```

```
"answer": "**RAG(Retrieval Augmented Generation)**에 대해 설명드리겠습니다:

1. **개요**:
   - RAG는 Retrieval Augmented Generation의 약자로, 정보 검색을 강화한 생성 모델을 가리킵니다.

2. **기술적 측면**:
   - RAG는 주로 세 가지 기본 구성 요소로 이루어져 있습니다: 정보 검색(Re-trieval), 생성(Generation), 그리고 조합(Combination).
   - 정보 검색 단계에서는 대규모 텍스트 데이터베이스로부터 관련 정보를 검색합니다.
   - 생성 단계에서는 검색된 정보를 바탕으로 자연스러운 문장 또는 텍스트를 생성합니다.
   - 조합 단계에서는 정보 검색과 생성을 효과적으로 결합하여 높은 품질의 결과물을 얻습니다.
  
3. **적용 분야**:
   - RAG는 자연어 이해, 대화형 인터페이스, 정보 검색 및 요약, 문서 생성 등 다양한 분야에 적용되고 있습니다.
   
4. **장점**:
   - RAG는 대량의 데이터를 기반으로 학습하므로 일반적으로 높은 성능을 보입니다.
   - 정보 검색과 생성을 융합함으로써 상호 보완적인 기능을 제공합니다.

5. **활용**:
   - RAG는 다양한 자연어 처리 작업에 유용하게 활용될 수 있으며, 특히 정보 검색 및 요약에 큰 도움을 줄 수 있습니다.
   
이러한 방식으로 RAG는 정보 검색과 생성을 결합하여 효율적이고 정확한 결과물을 제공하는 초거대 언어모델의 한 형태로 주목받고 있습니다."
```

[¶ Top](#index)
><br>

## 최종 작동 평가

### 평가 환경
- **Jupyter Notebook**
- **VSCode**

### 패키지 관리
- `ipywidgets-8.1.5` : **Jupyter Notebook** 에서 입력 기능 설정

### RAG Chain 코드 개선

#### 전달 함수 수정
- `context` 전달할 때 줄바꿈 제거

```py
class SimplePassThrough:
    def invoke(self, inputs, **kwargs):
        return inputs


class ContextToPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def invoke(self, inputs):
        # 문서 내용을 텍스트로 변환
        if isinstance(inputs, list):
            context_text = [doc.page_content.replace("\n", " ") for doc in inputs]
            
        else:
            context_text = inputs

        # 프롬프트 템플릿에 적용
        formatted_prompt = self.prompt_template.format_messages(
            context=context_text, question=inputs.get("question", "")
        )
        return formatted_prompt


# Retriever를 invoke() 메서드로 래핑하는 클래스 정의
class RetrieverWrapper:
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, inputs):
        if isinstance(inputs, dict):
            query = inputs.get("question", "")
        else:
            query = inputs
        # 검색 수행
        response_docs = self.retriever.invoke(query)
        return response_docs
```

### Lang Chain 생성 수정
- 외부에서도 프롬프트와 참고문서를 입력 받을 수 있게 수정

```py
from langchain.chains import LLMChain

PROMPT_BASELINE = "Answer the question using only the following context."
REFERENCE_BASELINE = "check user qestion"

def RAGChainMake(VECTOR_STORE, PARAMS, PROMPT=PROMPT_BASELINE, REFERENCE=REFERENCE_BASELINE, **kwargs):
    """
    RAG 기법을 이용한 대화형 LLM 답변 체인 생성 (히스토리 기억 및 동적 대화 기능 포함)

    VECTOR_STORE : Retriever가 검색할 벡터 스토어
    PARAMS       : API Key 및 LLM 모델명 등의 환경 변수 포함
    PROMPT       : 시스템 초기 프롬프트 (기본값 설정)
    REFERENCE    : 추가 문맥 정보 (선택 사항)
    """
    # 벡터 스토어에서 유사한 문맥 검색
    retriever = VECTOR_STORE.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}
    )

    # API 키 설정
    openai.api_key = os.environ.get(PARAMS.KEY)
    llm_model = ChatOpenAI(
        model=PARAMS.LLM_MODEL,
        api_key=openai.api_key,
    )

    # 대화형 프롬프트 생성
    contextual_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f'{PROMPT} \n\n reference : {REFERENCE}'),
            ("user", "Context: {context}\n\nQuestion: {question}"),
        ]
    )

    # RAG 체인 설정
    rag_chain_debug = {
        "context": RetrieverWrapper(retriever),
        "prompt": ContextToPrompt(contextual_prompt),
        "llm": llm_model,
    }

    return rag_chain_debug
```
### 대화형 답변 생성 기능
- 사용자의 입력을 받아 답변 진행
- 대화 내용을 기록
- 대화가 종료되면 저장 여부를 확인하여 저장

```py
from RAG_Module.Prompt_Engineering import *

def RAG_Coversation(CHAIN, PARAMS, **kwargs):
    """
    사용자로부터 질문을 받아 RAG 체인 기반으로 답변을 생성하는 대화형 함수
    전체 대화 결과를 리스트에 저장
    PARMS에 프롬프트 사용 및 결과 저장을 위한 PromptParams 입력
    """
    print("대화를 시작합니다. 종료하려면 'exit'를 입력하세요.\n")

    conversation_history = []  # 대화 기록을 저장할 리스트

    while True:
        print("========================")
        query = input("질문을 입력하세요 : ")

        if query.lower() == "exit":
            print("대화를 종료합니다.")
            break

        # 1. Retriever로 관련 문서 검색
        response_docs = CHAIN["context"].invoke({"question": query})

        # 2. 문서를 프롬프트로 변환
        prompt_messages = CHAIN["prompt"].invoke(
            {"context": response_docs, "question": query}
        )

        # 3. LLM으로 응답 생성
        response = CHAIN["llm"].invoke(prompt_messages)

        print("\n답변:")
        print(response.content)

        conversation_history.append({"question": query, "answer": response.content})

    while True:
        save_result = input("\n결과를 저장하시겠습니까? (y/n): ").strip().lower()

        if save_result == "y":
            PromptResult(conversation_history, PARAMS, **kwargs)
            print("결과가 저장되었습니다.")
            break  
        elif save_result == "n":
            print("결과가 저장되지 않았습니다. 대화를 종료합니다.")
            break  
        else:
            print(
                "잘못된 입력입니다. 다시 입력해주세요."
            ) 
```

### Chatbot 작동 테스트

```py
from RAG_Module.RAG_Params import *
from RAG_Module.PDF_Loader import PDFLoader
from RAG_Module.VecotorStore_Utils import VectorStoreReturn
from RAG_Module.RAG_Chain import *
from RAG_Module.Prompt_Engineering import *

# RAG 구성을 위한 파라미터
rag_setting = RAGParams(
    KEY="MY_OPENAI_API_KEY",
    EBD_MODEL="text-embedding-ada-002",
    LLM_MODEL="gpt-3.5-turbo",
    PDF_PATH="documents/초거대 언어모델 연구 동향.pdf",
    SAVE_PATH=None,
    IS_SAFE=True,
    CHUNK_SIZE=500,
    CHUNK_OVERLAP=100,
)

# 프롬프트 작성 및 사용을 위한 파라미터
prompt_setting = PromptParams(
    KEY="MY_OPENAI_API_KEY",
    LLM_MODEL="gpt-3.5-turbo",
    PROMPT_PATH="Prompts/",
    PROMPT_NAME="test_prompt",
    PROMPT_EXTENSION="txt",
    RESULT_PATH="Results/",
    RESULT_EXTENSION="txt",
)

# 문서 분할 및 벡터 스토어 생성
docs = PDFLoader(rag_setting)
vector_store = VectorStoreReturn(docs, rag_setting)

# 프롬프트 템플릿 작성
template_setting = TemplateParams(
    PERSONA="specialist of large language model",
    LANG="only in korean",
    TONE="professional",
    PERPOSE="study large language model",
    HOW_WRITE="itemization",
    CONDITION="""

    answer is about large language model
    prioritize the context of the question
    specify if there are any new information
    If you can identify the original source of the document you referenced, write in APA format

    """,
    REFERENCE= "context given in the question",
)

# 프롬프트 생성
prompt = PromptTemplate(template_setting)

# 챗봇 생성
## 벡터 스토어, RAG구성 파라미터, 프롬프트 입력
chatbot_mk3 = RAGChainMake(vector_store, rag_setting, prompt)

# 챗봇과 대화 시작
RAG_Coversation(chatbot_mk3, prompt_setting)
```

### 답변 비교
- 질문 : 업스테이지의 solar 모델에 대해 설명해줘.
- 확인 사항 : solar 모델은 2023년에 개발된 업스테이지의 최신 모델

#### GPT-3.5-turbo
- 최신 정보를 반영하지 못하고 할루시네이션 발생

```
업스테이지의 Solar 모델은 한국의 여성 가수이자 댄서인 솔라(Solar)의 이름을 딴 제품입니다. Solar 모델은 업스테이지의 다양한 제품 중 하나로, 솔라의 이미지와 스타일을 반영하여 디자인되었습니다. 이 제품은 솔라의 개성을 살린 컬러나 디자인으로 구성되어 있으며, 솔라의 음악과 무대 퍼포먼스와 어울리는 메이크업을 연출할 수 있도록 제작되었습니다. Solar 모델은 솔라의 팬이나 메이크업을 즐기는 이들에게 특별한 즐거움을 주는 제품으로 소개되고 있습니다
```

#### GPT-4o-mini
- 최신 정보를 반영하여 solar 모델에 대한 정보를 제공
- 상세한 정보를 반영하고 있지 않고 대략적인 정보만 제공

```
업스테이지(Upstage)의 Solar 모델은 자연어 처리(NLP)와 관련된 AI 모델 중 하나로, 주로 대화형 AI 시스템이나 문서 이해, 질의응답 시스템 등 다양한 언어 기반 작업을 수행할 수 있는 고성능 언어 모델입니다. Solar 모델은 대규모 데이터셋을 기반으로 학습된 트랜스포머 아키텍처를 사용하며, 한국어와 영어를 포함한 다양한 언어에서 뛰어난 성능을 발휘할 수 있도록 최적화되어 있습니다.

Solar 모델의 특징:

대규모 언어 모델: 대량의 텍스트 데이터를 학습하여, 다양한 언어적 상황에서 의미를 파악하고 이해할 수 있습니다.
트랜스포머 기반: Transformer 아키텍처를 기반으로 하기 때문에 문맥을 잘 이해하고, 긴 텍스트나 복잡한 문장 구조도 효과적으로 처리할 수 있습니다.
다양한 활용 가능성: 문서 요약, 감정 분석, 기계 번역, 대화형 AI, 질의응답 시스템 등 여러 가지 언어적 작업에서 활용될 수 있습니다.
한국어 최적화: 업스테이지는 특히 한국어에 최적화된 모델을 제공하고 있어, 한국어 처리 성능이 뛰어나고, 한국어를 사용하는 다양한 실시간 응용 프로그램에 효과적입니다.
이 모델은 업스테이지의 다른 AI 시스템들과 함께 사용되며, 기업 고객이나 연구자들이 효율적으로 텍스트 기반의 문제를 해결하는 데 도움을 줍니다
```

#### RAG + GPT-3.5-turbo
- GPT-3.5-turbo에 없는 정보를 RAG 검색을 이용하여 답변
- 업스테이지에서 작성한 문서르 참고했기 떄문에 모델명 등의 상세한 정보를 제공

```
- 업스테이지의 Solar 모델은 Llama2를 파인튜닝하여 개발된 Solar-0-70b 모델입니다.
- Solar 모델은 글로벌 LLM 플랫폼 중 하나인 Poe.com에서 서비스되고 있습니다.
- Solar 모델은 한국어와 영어를 모두 지원하고 있습니다."
```

### 결론
> RAG를 이용하면 찾는 정보가 LLM의 데이터에 없는 경우나 있더라도 상세한 내용을 설명하지 못하는 경우에 보다 자세하고 신뢰할 수 있는 답변을 생성 할 수 있다.

### 복수의 프롬프트 결과 비교

#### 파라미터 설정

```py
rag_setting = RAGParams(
    KEY="MY_OPENAI_API_KEY",
    EBD_MODEL="text-embedding-ada-002",
    LLM_MODEL="gpt-3.5-turbo",
    PDF_PATH="documents/초거대 언어모델 연구 동향.pdf",
    SAVE_PATH=None,
    IS_SAFE=True,
    CHUNK_SIZE=500,
    CHUNK_OVERLAP=100,
)

prompt_setting = PromptParams(
    KEY="MY_OPENAI_API_KEY",
    LLM_MODEL="gpt-4o-mini",
    PROMPT_PATH="Prompts/",
    PROMPT_NAME=None,
    PROMPT_EXTENSION="txt",
    RESULT_PATH="Results/",
    RESULT_EXTENSION="txt",
)
```

#### 벡터 스토어 생성 

```py
# 문서 불러오기 및 분할
docs = PDFLoader(rag_setting)

# 벡터 스토어 생성
vector_store = VectorStoreReturn(docs, rag_setting)
```

#### 프롬프트 1 작성 및 저장

```
prompt_1 = "Answer the question using only the following context."
PromptSave(prompt_1, prompt_setting, PROMPT_NAME='prompt_1')
```

#### 프롬프트 2 작성 및 저장

```
prompt_2 = """

you are specialist of large language model
answer question
refer to context qiven in question

"""
PromptSave(prompt_2, prompt_setting, PROMPT_NAME='prompt_2')
```

#### 프롬프트 3 작성 및 저장

- shot 기법 사용을 위한 shot 제작용 프롬프트 생성

```
shot_template = TemplateParams(
    PERSONA="specialist of large language model",
    LANG="only in korean",
    TONE="professional",
    PERPOSE="study large language model",
    HOW_WRITE="itemization",
    CONDITION="""

    <must obey>
    answer is about large language model
    answer that you do not know what you do not know
    </must obey>

    if you canspecify the date standard of the information
    if you can identify the original source of the document you referenced, write in APA format
    """,
    REFERENCE="only the latest information")
shot_prompt = PromptTemplate(shot_template)
```

- shot 생성 - 상위 모델인 gpt-4o-mini르 사용, 답변 방식을 참고하도록 유도

```
question = "gpt-4에 대해서 설명해줘"
shot = LLMSupport(question, prompt_setting, shot_prompt)
```

- 프롬프트 3 작성 및 저장

```
template_setting = TemplateParams(
    PERSONA="specialist of large language model",
    LANG="only in korean",
    TONE="professional",
    PERPOSE="study large language model",
    HOW_WRITE="itemization",
    CONDITION="""

    <must obey>
    answer is about large language model
    answer that you do not know what you do not know
    </must obey>

    prioritize the context in question
    specify if there are any new information

    if you can identify the original source of the document you referenced, write in APA format
    """,
    REFERENCE=f"""

    <answer format sample>
    {shot}
    </answer format>
    
    refer to context given in the question",
    """
)
prompt_3 = PromptTemplate(template_setting)
PromptSave(prompt_3, prompt_setting, PROMPT_NAME='prompt_3')
```

#### 저장된 프롬프트 1, 2, 3 불러오기 및 결과 저장
- `QUESTION` : 리스트 형태로 입력

```py
def AutoChain(PARAMS, VECTOR_STORE, QESTION, **kwargs):
    """
    미리 설정된 질문을 바탕으로 대화 없이 결과를 저장하는 함수
    폴더 내 모든 프롬프트 파일에 대해 실행하고 결과를 저장
    QUESTION 은 리스트 형태로 주어져야 함
    """

    # 1. 프롬프트 폴더 내의 모든 프롬프트 파일을 불러오기
    prompt_files = [f for f in os.listdir(PARAMS.PROMPT_PATH) if f.endswith(f".{PARAMS.PROMPT_EXTENSION}")]
    
    if not prompt_files:
        print("프롬프트 파일이 없습니다. 종료합니다.")
        return

    # 각 프롬프트 파일에 대해 처리
    for prompt_file in prompt_files:
        print(f"\n{prompt_file} 로 시작합니다.")

        # 2. 프롬프트 불러오기
        prompt_name = prompt_file.split('.')[0]  # 확장자를 제외한 파일 이름
        prompt = PromptLoad(PARAMS, PROMPT_NAME = prompt_name)
        if not prompt:
            print(f"{prompt_file} 로드 실패. 스킵합니다.")
            continue  # 실패하면 해당 프롬프트를 건너뜁니다.

        # 3. RAG 체인 만들기
        chain = RAGChainMake(VECTOR_STORE, PARAMS, PROMPT=prompt, **kwargs)

        conversation_history = []  # 대화 기록을 저장할 리스트

        for query in QESTION:
            print(f"질문: {query}")

            # 1. Retriever로 관련 문서 검색
            response_docs = chain["context"].invoke({"question": query})

            # 2. 문서를 프롬프트로 변환
            prompt_messages = chain["prompt"].invoke(
                {"context": response_docs, "question": query}
            )

            # 3. LLM으로 응답 생성
            response = chain["llm"].invoke(prompt_messages)

            print("답변:", response.content)

            conversation_history.append({"question": query, "answer": response.content})

        # 결과 저장
        PromptResult(conversation_history, PARAMS, PROMPT_NAME = prompt_name, **kwargs)
        print(f"{prompt_file}에 대한 결과 저장 완료.")
    
    print("모든 결과가 저장되었습니다.")
```

### 최종 결과
> RAG로 정보를 제공하고 상대적으로 고성능 모델의 답변 방식을 예시로 제공하면<br>
>저성능 모델에서도 의미있는 답변을 얻을 수 있음을 확인

#### question
```
 업스테이지의 solar 모델에 대해 설명해줘
```

#### prompt 1
- prompt

```
Answer the question using only the following context.
```

- answer

```
"업스테이지는 Llama2를 파인튜닝하여 Solar-0-70b 모델을 개발하였으며, 이 모델은 한국어와 영어 모두 지원하는 글로벌 LLM 플랫폼 중 하나인 Poe.com에서 서비스되고 있습니다."
```

#### prompt 2
- prompt

```
you are specialist of large language model
answer question
refer to context qiven in question
```

- answer

```
"업스테이지의 Solar-0-70b 모델은 Llama2를 파인튜닝하여 개발된 한국어 LLM입니다. 이 모델은 한국어와 영어를 모두 지원하며, 글로벌 LLM 플랫폼 중 하나인 Poe.com에서 서비스되고 있습니다."
```

#### prompt 3
- prompt

```
    persona : specialist of large language model
    language : only in korean
    tone : professional
    purpose : study large language model
    how to write : itemization
    condition : 

    <must obey>
    answer is about large language model
    answer that you do not know what you do not know
    </must obey>

    prioritize the context in question
    specify if there are any new information

    if you can identify the original source of the document you referenced, write in APA format
    
    reference : 

    <answer format sample>
    GPT-4에 대한 설명은 다음과 같습니다:

   1. **모델 개요**:
      - GPT-4는 OpenAI에서 개발한 대규모 언어 모델로, 자연어 처리(NLP) 작업을 수행하는 데 사용됩니다.
      - 이전 버전인 GPT-3에 비해 더 많은 파라미터와 개선된 알고리즘을 통해 성능이 향상되었습니다.

   2. **기능**:
      - 텍스트 생성: 주어진 프롬프트에 따라 자연스러운 문장을 생성할 수 있습니다.
      - 질문 응답: 사용자의 질문에 대해 관련 정보를 바탕으로 답변을 제공합니다.
      - 번역: 여러 언어 간의 번역 작업을 수행할 수 있습니다.
      - 요약: 긴 텍스트를 간결하게 요약하는 기능을 갖추고 있습니다.

   3. **훈련 데이터**:
      - GPT-4는 다양한 출처의 대규모 텍스트 데이터로 훈련되었습니다. 이는 웹사이트, 책, 논문 등 다양한 형식의 데이터를 포함합니다.
      - 훈련 데이터는 2023년 10월까지의 정보로 제한되어 있습니다.

   4. **응용 분야**:
      - 고객 서비스: 챗봇 및 가상 비서로 활용됩니다.
      - 콘텐츠 생성: 블로그, 기사, 소설 등 다양한 콘텐츠를 생성하는 데 사용됩니다.
      - 교육: 학습 도구로 활용되어 학생들에게 정보를 제공합니다.

   5. **한계**:
      - GPT-4는 훈련 데이터에 기반하여 작동하므로, 최신 정보나 특정 전문 분야에 대한 깊이 있는 지식이 부족할 수 있습니다.
      - 모델이 생성하는 정보의 정확성이나 신뢰성은 항상 보장되지 않으며, 사용자는 결과를 비판적으로 평가해야 합니다.

   6. **미래 전망**:
      - GPT-4와 같은 모델은 지속적으로 발전하고 있으며, 향후 더 나은 성능과 다양한 기능을 갖춘 모델이 개발될 것으로 예상됩니다.

   이 정보는 OpenAI의 공식 발표 및 연구 자료를 바탕으로 작성되었습니다. (OpenAI, 2023)
    </answer format>
    
    refer to context given in the question",
```

- answer

```
"업스테이지의 Solar 모델에 대한 설명은 다음과 같습니다:
1. **모델 개요**:
   - Solar 모델은 업스테이지에서 개발한 대규모 언어 모델입니다.
   - Llama2를 기반으로 파인튜닝되어 생성되었습니다.
      
2. **지원 언어**:
   - Solar 모델은 한국어와 영어를 모두 지원합니다.

3. **응용 분야**:
   - 해당 모델은 글로벌 LLM 플랫폼인 Poe.com에서 서비스되고 있습니다.
   
4. **훈련 데이터**:
   - 한국어 데이터와 공개된 한국어 데이터, 크롤링 데이터를 활용하여 학습하였습니다.
   
5. **모델의 특징**:
   - 한국어 토큰 비율을 높여 한국어 처리 성능을 개선하는 데 중점을 두고 있습니다.
   
이 정보는 제공된 문서의 내용을 바탕으로 작성되었습니다. (초거대 언어모델 연구 동향.pdf, p. 3)"
```

[¶ Top](#index)
><br>
