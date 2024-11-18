import os
import openai
from langchain_openai import OpenAIEmbeddings

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from uuid import uuid4

def VectorStoreReturn(SPLITS, PARAMS, **kwargs):
    """
    Document 객체를 임베딩하여 Vector Store 반환
    SPLITS : Document 객체
    PARAMS : RAGParams 객체
    KEY : 환경변수에서 호출할 API Key 이름
    EBD_MODEL : 임베딩 모델명
    """
    print('Authenticate API KEY....')
    openai.api_key = os.environ.get(PARAMS.KEY)
    print('Authenticate Complete....!')

    print('Set up Embedding model....')
    embedding_model = OpenAIEmbeddings(model=PARAMS.EBD_MODEL, api_key=openai.api_key)
    print('Set up Complete....!')

    print('Initiate FAISS instance....')
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("This is Sample Text.")))
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(SPLITS))]
    vector_store.add_documents(documents=SPLITS, ids=uuids)
    print('Return Vector Store....!\n')
    return vector_store

def VectorStoreSave(SPLITS, PARAMS, **kwargs):
    """
    Document 객체를 임베딩하여 Vector Store 저장
    SPLITS : Document 객체
    PARAMS : RAGParams 객체
    KEY : 환경변수에서 호출할 API Key 이름
    EBD_MODEL : 임베딩 모델명
    SAVE_PATH : 저장할 경로
    """
    print('Authenticate API KEY....')
    openai.api_key = os.environ.get(PARAMS.KEY)
    print('Authenticate Complete....!')

    embedding_model = OpenAIEmbeddings(model=PARAMS.EBD_MODEL, api_key=openai.api_key)
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("This is Sample Text.")))
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(SPLITS))]
    vector_store.add_documents(documents=SPLITS, ids=uuids)
    vector_store.save_local(PARAMS.SAVE_PATH)
    print('Vector Store Saved....!\n')

def VectorStoreLoad(PARAMS, **kwargs):
    """
    저장된 Vector Store 반환
    SPLITS : Document 객체
    PARAMS : RAGParams 객체
    KEY : 환경변수에서 호출할 API Key 이름
    EBD_MODEL : 임베딩 모델명
    SAVE_PATH : Vector Store가 저장된 경로
    IS_SAFE : 불러올 Vector Store이 안전한 파일인지 확인(불리언)
    """
    # API 키 인증
    print('Authenticate API KEY....')
    openai.api_key = os.environ.get(PARAMS.KEY)
    print('Authenticate Complete....!')

    # 임베딩 모델 설정
    print('Set up Embedding model....')
    print(f'model {PARAMS.EBD_MODEL} load....')
    embedding_model = OpenAIEmbeddings(
        model=PARAMS.EBD_MODEL, api_key=openai.api_key
    )
    print('Set up Complete....!')

    # 벡터 스토어 로드
    print('Load Vector Store....')
    vector_store = FAISS.load_local(
        PARAMS.SAVE_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=PARAMS.IS_SAFE
    )
    print('Return Vector Store....!\n')
    
    return vector_store