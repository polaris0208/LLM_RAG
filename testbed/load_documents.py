from langchain_community.document_loaders import PyPDFLoader

# 가져올 pdf 파일 주소
path = "documents/초거대 언어모델 연구 동향.pdf"

# 사용할 pdf loader 선택
loader = PyPDFLoader(path)

# pdf 파일 불러오기
docs = loader.load()

# 불러올 범위 설정
page = 0
start_point = 0
end_point = 500

# 결과 확인
print(f"페이지 수: {len(docs)}")
print(f"\n[페이지내용]\n{docs[page].page_content[start_point:end_point]}")
print(f"\n[metadata]\n{docs[page].metadata}\n")
