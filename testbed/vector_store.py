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

from langchain_openai import ChatOpenAI

# 환경 변수에서 호출
openai.api_key = os.environ.get("________")


model = ChatOpenAI(model="gpt-4", api_key=openai.api_key)

from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=openai.api_key
)

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

from uuid import uuid4

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
