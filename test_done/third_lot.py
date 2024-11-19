from RAG_Module.RAG_Params import *
from RAG_Module.PDF_Loader import PDFLoader
from RAG_Module.VecotorStore_Utils import VectorStoreReturn
from RAG_Module.RAG_Chain import *
from RAG_Module.Prompt_Engineering import *

rag_setting = RAGParams(
    KEY="MY_OPENAI_API_KEY",
    EBD_MODEL="text-embedding-ada-002",
    LLM_MODEL="gpt-3.5-turbo",
    PDF_PATH="documents/초거대 언어모델 연구 동향.pdf",
    SAVE_PATH=None,
    IS_SAFE=True,
    CHUNK_SIZE=100,
    CHUNK_OVERLAP=10,
)

prompt_setting = PromptParams(
    KEY="MY_OPENAI_API_KEY",
    LLM_MODEL="gpt-3.5-turbo",
    PROMPT_PATH="Prompts/",
    PROMPT_NAME="test_prompt",
    PROMPT_EXTENSION="txt",
    RESULT_PATH="Results/",
    RESULT_EXTENSION="txt",
)

docs = PDFLoader(rag_setting)
vector_store = VectorStoreReturn(docs, rag_setting)
question = "업스테이지의 solar 모델에 대해 설명해줘."
shot = LLMSupport(question, prompt_setting)


template_setting = TemplateParams(
    PERSONA="specialist of large language model",
    LANG="only in korean",
    TONE="professional",
    PERPOSE="study large language model",
    HOW_WRITE="itemization",
    CONDITION="""

    answer is about large language model
    prioritize the context of the question
    specify if there are any new information
    If you can identify the original source of the document you referenced, write in APA format

    """,
    REFERENCE=f"{shot} \n\n and context given in the question",
)

prompt = PromptTemplate(template_setting)
chatbot_mk3 = RAGChainMake(vector_store, rag_setting, prompt)
RAG_Coversation(chatbot_mk3, prompt_setting)

print(shot)

"""답변:
업스테이지의 Solar 모델은 Llama2를 파인튜닝하여 개발된 Solar-0-70b 모델을 의미합니다. 
이 모델은 업스테이지에서 솔라를 모델로 채택하여 제품을 소비자들에게 홍보하고 있습니다. 
Solar 모델은 솔라의 이미지와 스타일을 반영하여 다양한 화장품 제품들을 소비자들에게 소개하고 있습니다. 
솔라는 업스테이지와의 협업을 통해 자신의 아름다움과 매력을 더욱 돋보이게 하고, 제품의 매력을 소비자들에게 전달하고 있습니다.

업스테이지의 Solar 모델은 한국의 여성 가수이자 댄서인 솔라(Solar)를 모델로 한 제품입니다. 
Solar 모델은 업스테이지의 다양한 화장품 제품들을 홍보하고 있으며, 솔라의 이미지와 스타일을 반영하여 제품을 소비자들에게 소개하고 있습니다. 
솔라는 업스테이지와의 협업을 통해 자신의 아름다움과 매력을 더욱 돋보이게 하고, 제품의 매력을 소비자들에게 전달하고 있습니다."""