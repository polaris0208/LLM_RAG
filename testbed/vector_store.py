from langchain_community.document_loaders import PyPDFLoader

path = "documents/초거대 언어모델 연구 동향.pdf"

loader = PyPDFLoader(path)

docs = loader.load()

# 파라미터 설정- 축소 테스트
CHUNK_INDEX = 0
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서 분할기 설정
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

# 문서 분할
splits = recursive_splitter.split_documents(docs)

###################################################################### 전처리 과정

import os
import openai

# 환경 변수에서 호출
openai.api_key = os.environ.get("NBCAMP_01")

from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=openai.api_key
)

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

index = faiss.IndexFlatL2(len(embedding_model.embed_query("This is Sample Text.")))
vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

from uuid import uuid4

# for split in splits:
#     split.metadata["uuid"] = str(uuid4())
# vector_store.add_documents(documents=documents)

uuids = [str(uuid4()) for _ in range(len(splits))]
vector_store.add_documents(documents=splits, ids=uuids)

# 테스트
results_with_scores = vector_store.similarity_search_with_score(
    "RAG에 대해 이야기해주세요.", k=2, filter={"source": 'documents/초거대 언어모델 연구 동향.pdf'}
)
for res, score in results_with_scores:
    print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")

# 결과
"""
* [SIM=0.345] 16 특집원고  초거대 언어모델 연구 동향
Retrieval Augmented Generation (RAG) [95, 96, 97, 
98]이라 한다.
Other Tools L a M D A  [ 4 4 ]는 대화 애플리케이션에 
특화된 LLM으로, 검색 모듈, 계산기 및 번역기 등의 
외부 도구 호출 기능을 가지고 있다. WebGPT [99] 는 
웹 브라우저와의 상호작용을 통해 검색 쿼리에 사실 
기반의 답변과 함께 출처 정보를 제공 한다. PAL 
[100]은 Python 인터프리터를 통한 복잡한 기호 추론 
기능을 제공하며, 여러 관련 벤치마크에서 뛰어난 성
능을 보여주었다. 다양한 종류의 API (e.g., 계산기, 달
력, 검색, QA, 번역 등 단순한 API에서부터 Torch/ 
TensorFlow/HuggingFace Hub에 이르는 복잡한 API까
지) 호출 기능을 갖춘 연구들 [101, 102, 103, 104, [{'source': 'documents/초거대 언어모델 연구 동향.pdf', 'page': 8}]

* [SIM=0.424] 같은 문자라면 같은 밀집 벡터로 표현하기 때문에 문
맥 정보를 반영하지 못한다는 한계를 지닌다.
문맥기반 언어모델 연구 문맥 정보를 반영하여 언
어를 표현하기 위해, 텍스트 내의 정보를 이용하는 
RNN (Recurrent Neural Network) 이 등장했다. 그러나, 
RNN은 입력 텍스트의 길이가 길어질수록 앞쪽에 위 [{'source': 'documents/초거대 언어모델 연구 동향.pdf', 'page': 1}]
"""

# 저장

vector_store.save_local("testbed/vector_store_index/faiss_index")