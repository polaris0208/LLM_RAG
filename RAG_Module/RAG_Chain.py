import os
import openai
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


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

def RAGChainMake(VECTOR_STORE, PARAMS, **kwargs):
  """
  RAG 기법을 이용한 LLM 답변 구조 생성
  VECTOR_STORE : Retriever가 검색할 벡터 스토어
  KEY : 환경변수에서 호출할 API Key 이름
  LLM_MODEL : 사용할 LLM 모델명
  """
  retriever = VECTOR_STORE.as_retriever(
      search_type="similarity", search_kwargs={"k": 1}
      )
  
  openai.api_key = os.environ.get(PARAMS.KEY)
  llm_model = ChatOpenAI(model=PARAMS.LLM_MODEL, api_key=openai.api_key)

  contextual_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question using only the following context."),
            ("user", "Context: {context}\\n\\nQuestion: {question}"),
        ]
    )
  rag_chain_debug = (
    {"context": retriever, "question": DebugPassThrough()}
    | DebugPassThrough()
    | ContextToText()
    | contextual_prompt
    | llm_model
    )
  return rag_chain_debug