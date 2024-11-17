from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def PDFLoader(PARAMS, **kwargs):
    """
    PDF 파일을 입력 받아 Document 객체 반환

    PARAMS: RAGParams 객체
    PDF_PATH : 사용 pdf 파일 경로
    CHUNK_SIZE : 청크당 문자 수 
    CHUNK_OVERLAP : 중첩시킬 문자 수
    """
    loader = PyPDFLoader(PARAMS.PDF_PATH)
    
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARAMS.CHUNK_SIZE,
        chunk_overlap=PARAMS.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    print("\nLoad PDF.....")
    docs = loader.load()
    print("Load Complete....!")

    print("Split docs.....")
    splits = recursive_splitter.split_documents(docs)
    print("Split Complete....!\n")

    return splits