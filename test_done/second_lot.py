from RAG_Module.RAG_Params import RAGParams
from RAG_Module.PDF_Loader import PDFLoader
from RAG_Module.VecotorStore_Utils import VectorStoreReturn
from RAG_Module.RAG_Chain_exhausted import RAGChainMake

params = RAGParams(
    KEY= "MY_OPENAI_API_KEY",
    EBD_MODEL="text-embedding-ada-002",
    LLM_MODEL='gpt-3.5-turbo',
    PDF_PATH="documents/초거대 언어모델 연구 동향.pdf",
    SAVE_PATH = None,
    IS_SAFE=True,
    CHUNK_SIZE = 1200,
    CHUNK_OVERLAP = 300
)

docs = PDFLoader(params)
vector_store = VectorStoreReturn(docs, params)
chatbot_mk1 = RAGChainMake(vector_store, params)

question = "업스테이지의 solar 모델에 대해 설명해줘."
response = chatbot_mk1.invoke(question)

print(response.content)

"""Upstage의 solar 모델은 해당 분야에서 매우 높은 성과를 보여주고 있으며, 
많은 기업들이 이에 참여하고 있다. 이 모델은 오픈 초기에는 평균 점수가 30점대 초반이었지만,
 2주만에 대부분 45점을 돌파하여 50%의 큰 향상폭을 보였습니다. 
 따라서 solar 모델은 치열한 경쟁과 다양한 모델들의 활발한 참여로 인해 주목받고 있는 것으로 알려져 있습니다."""