import os
import openai
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain

from RAG_Module.Prompt_Engineering import *


PROMPT_BASELINE = "Answer the question using only the following context."
REFERENCE_BASELINE = "check user qestion"

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


def RAGChainMake(VECTOR_STORE, PARAMS, PROMPT=PROMPT_BASELINE, REFERENCE=REFERENCE_BASELINE, **kwargs):
    """
    RAG 기법을 이용한 대화형 LLM 답변 체인 생성 (히스토리 기억 및 동적 대화 기능 포함).

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


def RAG_Coversation(CHAIN, PARAMS, **kwargs):
    """
    사용자로부터 질문을 받아 RAG 체인 기반으로 답변을 생성하는 대화형 함수.
    전체 대화 결과를 리스트에 저장.
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
