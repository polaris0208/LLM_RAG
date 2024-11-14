# pdf 문서 로드
from langchain_community.document_loaders import PyPDFLoader

path = "documents/초거대 언어모델 연구 동향.pdf"

loader = PyPDFLoader(path)

docs = loader.load()

######################################################### 이전 단계

# 파라미터 설정
CHUNK_INDEX = 0
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
SEPERATOR = "\n"

from langchain.text_splitter import CharacterTextSplitter

# 문서 분할기 설정
splitter = CharacterTextSplitter(
    separator=SEPERATOR,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=True,
)

# 문서 분할
c_splits = splitter.split_documents(docs)

# 결과 확인
print(
    f"c_splits\n길이 : {len(c_splits)}\n 결과 확인 : {c_splits[CHUNK_INDEX].page_content}"
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서 분할기 설정
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

# 문서 분할
r_splits = recursive_splitter.split_documents(docs)

# 결과 확인
print(
    f"r_splits\n길이 : {len(r_splits)}\n 결과 확인 : {r_splits[CHUNK_INDEX].page_content}"
)


# Sample 10개 테스트

def sample_printer(splits_1, splits_2, n):
    for i in range(n):
        print(f"Sample 생성중...")
        print(f"Sample_1 {i} \n {splits_1[i].page_content}")
        print(f"Sample_2 {i} \n {splits_2[i].page_content}")


# sample_printer(c_splits, r_splits, 10)
