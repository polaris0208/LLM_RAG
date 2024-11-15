import os
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

openai.api_key = os.environ.get("NBCAMP_01")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=openai.api_key
)

vector_store = FAISS.load_local(
    "testbed/vector_store_index/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True,
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

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
        context_text = [doc.page_content.replace('\n', ' ') for doc in inputs["context"]]
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

response = rag_chain_debug.invoke("RAG에 대해 이야기해주세요.")
print("Final Response:")
print(response.content)


""" \n 제거전
Debug Output: RAG에 대해 이야기해주세요.
Debug Output: {'context': [Document(metadata={'source': 'documents/초거대 언어모델 연구 동향.pdf', 'page': 8}, page_content='16 특집원고  초거대 언어모델 연구 동향\nRetrieval Augmented Generation (RAG) [95, 96, 97, \n98]이라 한다.\nOther Tools L a M D A  [ 4 4 ]는 대화 애플리케이션에 \n특화된 LLM으로, 검색 모듈, 계산기 및 번역기 등의 \n외부 도구 호출 기능을 가지고 있다. WebGPT [99] 는 \n웹 브라우저와의 상호작용을 통해 검색 쿼리에 사실 \n기반의 답변과 함께 출처 정보를 제공 한다. PAL \n[100]은 Python 인터프리터를 통한 복잡한 기호 추론 \n기능을 제공하며, 여러 관련 벤치마크에서 뛰어난 성\n능을 보여주었다. 다양한 종류의 API (e.g., 계산기, 달\n력, 검색, QA, 번역 등 단순한 API에서부터 Torch/ \nTensorFlow/HuggingFace Hub에 이르는 복잡한 API까\n지) 호출 기능을 갖춘 연구들 [101, 102, 103, 104,')], 'question': 'RAG에 대해 이야기해주세요.'}
Final Response:
RAG는 Retrieval Augmented Generation의 약자로, 정보 검색 기능을 활용하여 생성 모델의 성능을 향상시키는 접근 방식입니다. 이 방법은 모델이 질문에 대한 답변을 생성할 때, 외부 데이터베이스나 검색 엔진에서 관련 정보를 검색한 후 이를 바탕으로 보다 정확하고 사실적인 응답을 생성할 수 있도록 돕습니다. RAG는 특히 대화형 AI나 질의응답 시스템에서 유용하게 사용됩니다.
"""

"""\n 제거
Debug Output: RAG에 대해 이야기해주세요.
Debug Output: {'context': [Document(metadata={'source': 'documents/초거대 언어모델 연구 동향.pdf', 'page': 8}, page_content='16 특집원고  초거대 언어모델 연구 동향\nRetrieval Augmented Generation (RAG) [95, 96, 97, \n98]이라 한다.\nOther Tools L a M D A  [ 4 4 ]는 대화 애플리케이션에 \n특화된 LLM으로, 검색 모듈, 계산기 및 번역기 등의 \n외부 도구 호출 기능을 가지고 있다. WebGPT [99] 는 \n웹 브라우저와의 상호작용을 통해 검색 쿼리에 사실 \n기반의 답변과 함께 출처 정보를 제공 한다. PAL \n[100]은 Python 인터프리터를 통한 복잡한 기호 추론 \n기능을 제공하며, 여러 관련 벤치마크에서 뛰어난 성\n능을 보여주었다. 다양한 종류의 API (e.g., 계산기, 달\n력, 검색, QA, 번역 등 단순한 API에서부터 Torch/ \nTensorFlow/HuggingFace Hub에 이르는 복잡한 API까\n지) 호출 기능을 갖춘 연구들 [101, 102, 103, 104,')], 'question': 'RAG에 대해 이야기해주세요.'}
Final Response:
RAG, 즉 Retrieval Augmented Generation은 초거대 언어모델 연구의 한 동향으로, 정보 검색과 생성 과정을 결합하여 보다 정확하고 정보에 기반한 응답을 생성하는 방법입니다. 이 접근 방식은 기존의 언어 모델이 가지고 있는 한계점을 극복하고, 외부 데이터 소스에서 정보를 검색하여 그에 기반한 답변을 제공할 수 있도록 합니다. RAG는 특히 다양한 API와의 통합을 통해 계산, 번역, 검색 등의 기능을 활용하여 더 나은 성능을 발휘할 수 있습니다.
"""
