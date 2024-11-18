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
    PROMPT_NAME=None,
    PROMPT_EXTENSION="txt",
    RESULT_PATH="Results/",
    RESULT_EXTENSION="txt",
)

docs = PDFLoader(rag_setting)
vector_store = VectorStoreReturn(docs, rag_setting)
question = "RAG에 대해서 설명해주세요"
shot = LLMSupport(question, prompt_setting)

prompt_1 = "Answer the question using only the following context."
PromptSave(prompt_1, prompt_setting, PROMPT_NAME='prompt_1')

prompt_2 = """

you are specialist of large language model
answer question
refer to context qiven in question

"""

PromptSave(prompt_2, prompt_setting, PROMPT_NAME='prompt_2')


template_setting = TemplateParams(
    PERSONA="specialist of large language model",
    LANG="korean",
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

prompt_3 = PromptTemplate(template_setting)
PromptSave(prompt_3, prompt_setting, PROMPT_NAME='prompt_3')

QUESTION = ["RAG에 대해서 설명해주세요",]
AutoChain(prompt_setting, vector_store, QUESTION)