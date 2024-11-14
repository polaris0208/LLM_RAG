from langchain_community.document_loaders import PyPDFLoader

path = "documents/초거대 언어모델 연구 동향.pdf"

loader = PyPDFLoader(path)

docs = loader.load()

# 파라미터 설정
CHUNK_INDEX = 0
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300

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

# pip install openai / openai-1.54.4

from langchain_openai import ChatOpenAI

# pip install langchain-openai / langchain-openai-0.2.8

openai.api_key = os.environ.get("________")
# 환경 변수에서 호출

model = ChatOpenAI(model="gpt-4", api_key=openai.api_key)

from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=openai.api_key
)

# pip install faiss-cpu / faiss-cpu-1.9.0
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# FAISS 객체 초기화 방식 - 더 많은 설정, 대용량 문서 처리가 요구되거나 성능을 최대한으로 끌어올려야 할 상황
# 인덱스나, 임베딩이 계산되어 있는 경우 빠르게 진행
#
# docstore :각 벡터와 연관된 문서 데이터를 메모리에 저장
# index_to_docstore_id: 특정 벡터가 어떤 문서와 연관되는지를 저장

index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# from_documents 방식 - 간단하고 빠름, 일반적인 상황
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

# pip install uuid / uuid-1.30
from uuid import uuid4

# 고유한 UUID(Universally Unique Identifier)를 생성
# 128비트 숫자로, 이를 통해 거의 중복 없이 고유한 식별자를 생성
# 여러 문서를 처리하거나 분석하는 과정에서, 각 문서를 쉽게 추적하고 연결하려면 고유 식별자가 중요

for split in splits:
    split.metadata["uuid"] = str(uuid4())
vector_store.add_documents(documents=documents)

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)

# 테스트
results_with_scores = vector_store.similarity_search_with_score(
    "LangChain에 대해 이야기해주세요.", k=2, filter={"source": "tweet"}
)
for res, score in results_with_scores:
    print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")


