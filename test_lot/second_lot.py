from RAG_Module.RAG_Params import RAGParams
from RAG_Module.PDF_Loader import PDFLoader
from RAG_Module.VecotorStore_Utils import VectorStoreReturn
from RAG_Module.RAG_Chain_exhausted import RAGChainMake

params = RAGParams(
    KEY= "NBCAMP_01",
    EBD_MODEL="text-embedding-ada-002",
    LLM_MODEL='gpt-3.5-turbo',
    PDF_PATH="documents/초거대 언어모델 연구 동향.pdf",
    SAVE_PATH = None,
    IS_SAFE=True,
    CHUNK_SIZE = 100,
    CHUNK_OVERLAP = 10
)

docs = PDFLoader(params)
vector_store = VectorStoreReturn(docs, params)
chatbot_mk1 = RAGChainMake(vector_store, params)

question = 'RAG에 대해서 설명해주세요'
response = chatbot_mk1.invoke(question)

print(response.content)