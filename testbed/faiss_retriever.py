from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 리트리버
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# 리트리버 테스트
query = "질문"

retriever = vectorstore.as_retriever(search_type="similarity")
search_result = retriever.get_relevant_documents(query)
print(search_result)