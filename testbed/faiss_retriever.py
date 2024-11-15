import os
import openai
from langchain_openai import OpenAIEmbeddings
# OpenAIEmbeddings 클래스는 LangChain 0.0.9 버전에서 더 이상 사용되지 않게 되었으며, 향후 1.0에서 제거될 예정

from langchain_community.vectorstores import FAISS

openai.api_key = os.environ.get("NBCAMP_01")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=openai.api_key
)
# allow_dangerous_deserialization=True  
# 역직렬화 허용 설정
# 역직렬화는 저장된 데이터를 다시 Python 객체로 복원
# 악의적인 사용자가 파일을 수정하여 pickle 파일을 로드할 때 악성 코드를 실행
# 신뢰할 수 있는 출처에서만 pickle 파일을 사용
# 인터넷에서 다운로드한 pickle 파일을 로드 주의
vector_store = FAISS.load_local("testbed/vector_store_index/faiss_index", embedding_model, allow_dangerous_deserialization=True)

# 리트리버
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# 리트리버 테스트
query = "RAG에 대해 이야기해주세요."

retriever = vector_store.as_retriever(search_type="similarity")
results = retriever.invoke(query)
for result in results:
    print(f"Source: {result.metadata['source']} | Page: {result.metadata['page']}")
    print(f"Content: {result.page_content.replace('\n', ' ')}\n")

# 결과
"""
Source: documents/초거대 언어모델 연구 동향.pdf | Page: 8
Content: 16 특집원고  초거대 언어모델 연구 동향
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
지) 호출 기능을 갖춘 연구들 [101, 102, 103, 104,

Source: documents/초거대 언어모델 연구 동향.pdf | Page: 1
Content: 같은 문자라면 같은 밀집 벡터로 표현하기 때문에 문
맥 정보를 반영하지 못한다는 한계를 지닌다.
문맥기반 언어모델 연구 문맥 정보를 반영하여 언
어를 표현하기 위해, 텍스트 내의 정보를 이용하는 
RNN (Recurrent Neural Network) 이 등장했다. 그러나, 
RNN은 입력 텍스트의 길이가 길어질수록 앞쪽에 위

Source: documents/초거대 언어모델 연구 동향.pdf | Page: 2
Content: 계를 그대로 가진다: 1) 하나의 벡터에 텍스트의 모든 
정보를 담기 때문에 정보 손실이 발생하고, 2) 입력 
텍스트의 길이가 길어지면 기울기 소실 (gradient 
vanishing)이 발생한다.
이러한 한계를 해결하기 위해 나온 것이 바로 
Attention Mechanism [2] 과 이를 활용한 Transformer 
Architecture [3] 이다. Attention Mechanism 은 하나의 
벡터에 텍스트의 모든 정보를 담는 RNN, LSTM, 
GRU와 다르게, 텍스트 내 단어들의 벡터들을 필요에 
따라 적절히 활용하는 메커니즘이다. 현재 언어모델
의 근간이 되는 Transformer가 바로 이러한 Atten- 
tion Mechanism을 기반으로 한다. Transformer는 크게 
인코더와 디코더로 구성되는데, 인코더는 주어진 텍
스트를 이해하는 역할을 하고 디코더는 이해한 텍스
트를 기반으로 언어를 생성해내는 역할을 수행한다.

Source: documents/초거대 언어모델 연구 동향.pdf | Page: 0
Content: 리 하드웨어의 개발은 모델 학습에 있어 병목 현상을 
크게 완화시켰다. 이로 인해 연구자들은 모델의 복잡
성을 키우고, 더욱 깊은 신경망 구조를 탐구할 수 있
게 되었다. 셋째, 알고리즘 및 기술의 발전은 LLM의 
성능 향상을 주도하였다. Attention 및 Transformer 
Architecture의 도입은 연구자들에게 문맥 간의 관계
를 더욱 정교하게 모델링할 수 있는 방법을 제공하였
다 [2, 3]. 이 모든 변화의 중심에는 ‘scaling law’라는 
* 정회원
1) https://openai.com/blog/chatgpt
학문적인 통찰이 있다 [4]. 해당 연구에 따르면, 모델
의 크기와 그 성능은 긍정적인 상관 관계를 보인다. 
이를 통해 연구자들은 모델의 파라미터 수를 증가시
키면서, 이에 따른 성능 향상을 기술적 진보의 상호 
작용에서 나온 결과이며, 이러한 추세는 앞으로도 
NLP 연구의 주요 동력이 될 것으로 예상된다.
"""

