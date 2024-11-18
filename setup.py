from setuptools import setup, find_packages

setup(
    name="RAG_Module",
    version="0.1.0",
    description="RAG 작동을 위한 모듈",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'faiss-cpu==1.9.0',
        'jsonpatch==1.33',
        'jsonpointer==3.0.0',
        'langchain==0.3.7',  
        'numpy==1.26.4',
        'openai==1.54.4',
        'pypdf==5.1.0',
    ],
)