"""문서 불러오기"""

from langchain_community.document_loaders import PyPDFLoader

# 가져올 pdf 파일 경로
path = "documents/초거대 언어모델 연구 동향.pdf"

# 사용할 pdf loader 선택
loader = PyPDFLoader(path)

# pdf 파일 불러오기
docs = loader.load()

# 불러올 범위 설정
page = 0
start_point = 0
end_point = 100

# 결과 확인
print("Load PDF.....\n")
print(f"페이지 수: {len(docs)}")
print(f"\n[페이지내용]\n{docs[page].page_content[start_point:end_point]}")
print(f"\n[metadata]\n{docs[page].metadata}")
print("\nLoad Complete....!\n")

"""문서 분할하기"""

# 파라미터 설정
CHUNK_INDEX = 0
CHUNK_SIZE = 100
CHUNK_OVERLAP = 10
SEPERATOR = "\n"

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

# 결과 확인
print("Split docs.....\n")
print(f"splits\n길이 : {len(splits)}\n 결과 확인 : {splits[CHUNK_INDEX].page_content}")
print("\nSplit Complete....!\n")

"""문서 임베딩"""

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

from uuid import uuid4

uuids = [str(uuid4()) for _ in range(len(splits))]
vector_store.add_documents(documents=splits, ids=uuids)

results_with_scores = vector_store.similarity_search_with_score(
    "RAG에 대해 이야기해주세요.",
    k=1,
    filter={"source": "documents/초거대 언어모델 연구 동향.pdf"},
)

print("Search in Vector Store.....\n")
for res, score in results_with_scores:
    print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")
print("\nSearch Complete....!\n")

"""리트리버"""

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# 리트리버 테스트
query = "RAG에 대해 이야기해주세요."

retriever = vector_store.as_retriever(search_type="similarity")
results = retriever.invoke(query)

print("Retriever runs.....\n")
for result in results:
    print(f"Source: {result.metadata['source']} | Page: {result.metadata['page']}")
    print(f"Content: {result.page_content.replace('\n', ' ')}\n")
print("Retriever is back....!\n")

"""랭체인"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

contextual_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only the following context."),
        ("user", "Context: {context}\\n\\nQuestion: {question}"),
    ]
)


class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        print("Debug Output:", output)
        return output


class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):
        context_text = [
            doc.page_content.replace("\n", " ") for doc in inputs["context"]
        ]
        return {"context": context_text, "question": inputs["question"]}


from langchain_openai import ChatOpenAI

llm_model = ChatOpenAI(model="gpt-4o-mini", api_key=openai.api_key)

rag_chain_debug = (
    {"context": retriever, "question": DebugPassThrough()}
    | DebugPassThrough()
    | ContextToText()
    | contextual_prompt
    | llm_model
)

print("RAG chain works.....\n")
response = rag_chain_debug.invoke("RAG에 대해 이야기해주세요.")
print("Final Response:")
print(response.content)
print("\ndone.....")

"""실행결과

Load PDF.....

페이지 수: 17

[페이지내용]
8 특집원고  초거대 언어모델 연구 동향
초거대 언어모델 연구 동향
업스테이지  박찬준*･이원성･김윤기･김지후･이활석
 
1. 서  론1)
ChatGPT1)와 같은 초거대 언어모델

[metadata]
{'source': 'documents/초거대 언어모델 연구 동향.pdf', 'page': 0}

Load Complete....!

Split docs.....

splits
길이 : 755
 결과 확인 : 8 특집원고  초거대 언어모델 연구 동향
초거대 언어모델 연구 동향
업스테이지  박찬준*･이원성･김윤기･김지후･이활석
 
1. 서  론1)

Split Complete....!

Search in Vector Store.....

* [SIM=0.287] 16 특집원고  초거대 언어모델 연구 동향
Retrieval Augmented Generation (RAG) [95, 96, 97, 
98]이라 한다. [{'source': 'documents/초거대 언어모델 연구 동향.pdf', 'page': 8}]

Search Complete....!

Retriever runs.....

Source: documents/초거대 언어모델 연구 동향.pdf | Page: 8
Content: 16 특집원고  초거대 언어모델 연구 동향 Retrieval Augmented Generation (RAG) [95, 96, 97,  98]이라 한다.

Source: documents/초거대 언어모델 연구 동향.pdf | Page: 7
Content: 해당 섹션에서는 도구 사용에 대해서 주로 논의할 것 이다. 이외의 Augmented LLMs에 대한 심도 있는 논 의는 다음 논문을 참조하길 바란다 [94].

Source: documents/초거대 언어모델 연구 동향.pdf | Page: 5
Content: 음을 시사한다. 보다 자세한 Instruction Tuning에 대한  논의는 다음의 서베이 논문들을 참고하길 바란다 [49,  56, 68].

Source: documents/초거대 언어모델 연구 동향.pdf | Page: 1
Content: 부족, 최신 정보를 반영하는 데의 한계 등 여러 문제 점이 있으며, 이러한 문제점들을 해결하는 것은 다가 오는 연구의 중요한 도전 과제로 여겨진다.

Retriever is back....!

RAG chain works.....

Debug Output: RAG에 대해 이야기해주세요.
Debug Output: {'context': [Document(metadata={'source': 'documents/초거대 언어모델 연구 동향.pdf', 'page': 8}, page_content='16 특집원고  초거대 언어모델 연구 동향\nRetrieval Augmented Generation (RAG) [95, 96, 97, \n98]이라 한다.'), Document(metadata={'source': 'documents/초거대 언어모델 연구 동향.pdf', 'page': 7}, page_content='해당 섹션에서는 도구 사용에 대해서 주로 논의할 것\n이다. 이외의 Augmented LLMs에 대한 심도 있는 논\n의는 다음 논문을 참조하길 바란다 [94].'), Document(metadata={'source': 'documents/초거대 언어모델 연구 동향.pdf', 'page': 5}, page_content='음을 시사한다. 보다 자세한 Instruction Tuning에 대한 \n논의는 다음의 서베이 논문들을 참고하길 바란다 [49, \n56, 68].'), Document(metadata={'source': 'documents/초거대 언어모델 연구 동향.pdf', 'page': 1}, page_content='부족, 최신 정보를 반영하는 데의 한계 등 여러 문제\n점이 있으며, 이러한 문제점들을 해결하는 것은 다가\n오는 연구의 중요한 도전 과제로 여겨진다.')], 'question': 'RAG에 대해 이야기해주세요.'}
Final Response:
RAG는 Retrieval Augmented Generation의 약자로, 초거대 언어모델 연구의 한 동향을 나타냅니다. 이 기술은 정보 검색과 생성 모델을 결합하여 보다 정확하고 유용한 정보를 제공하는 데 중점을 두고 있습니다. RAG는 최신 정보를 반영하는 데 한계가 있을 수 있으며, 이러한 문제를 해결하는 것이 향후 연구에서 중요한 도전 과제로 여겨집니다.

done.....
"""