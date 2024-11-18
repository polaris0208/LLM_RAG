import os
from datetime import datetime
import json

import openai
from openai import OpenAI


def PromptSave(PROMPT, PARAMS, PROMPT_NAME=None, **kwargs):
    """
    주어진 프롬프트를 파일에 저장하는 함수

    PROMPT_PATH: str  # 프롬프트 파일 경로
    PROMPT_NAME: str  # 프롬프트 파일 이름
    PROMPT_EXTENSION: str  # 프롬프트 파일 확장자

    PROMPT_NAME이 지정되지 않을 때 PARAMS.PROMPT_NAME을 사용.
    """
    # `prompt_name`이 주어지지 않으면 `PARAMS.PROMPT_NAME`을 사용
    if PROMPT_NAME is None:
        PROMPT_NAME = PARAMS.PROMPT_NAME
    file_name = f"{PROMPT_NAME}.{PARAMS.PROMPT_EXTENSION}"

    # 파일 경로 생성
    file_path = os.path.join(PARAMS.PROMPT_PATH, file_name)
    os.makedirs(PARAMS.PROMPT_PATH, exist_ok=True)

    # 파일 저장
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(PROMPT)

    print(f"{PROMPT_NAME} 이/가 {PARAMS.PROMPT_PATH} 에 저장되었습니다.\n")


def PromptLoad(PARAMS, PROMPT_NAME=None, **kwargs):
    """
    저장된 프롬프트를 불러오는 함수

    PROMPT_PATH: str  # 프롬프트 파일 경로
    PROMPT_NAME: str  # 프롬프트 파일 이름
    PROMPT_EXTENSION: str  # 프롬프트 파일 확장자

    PROMPT_NAME이 지정되지 않을 때 PARAMS.PROMPT_NAME을 사용.
    """
    # `prompt_name`이 주어지지 않으면 `PARAMS.PROMPT_NAME`을 사용
    if PROMPT_NAME is None:
        PROMPT_NAME = PARAMS.PROMPT_NAME

    file_name = f"{PROMPT_NAME}.{PARAMS.PROMPT_EXTENSION}"
    # 파일 이름과 확장자를 결합하여 파일 경로 생성
    file_name = f"{PROMPT_NAME}.{PARAMS.PROMPT_EXTENSION}"
    file_path = os.path.join(PARAMS.PROMPT_PATH, file_name)

    # 파일이 존재하는지 확인
    if os.path.exists(file_path):
        # 파일 불러오기
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()  # 파일 내용 읽기
        print(f"{PROMPT_NAME} 이/가 {PARAMS.PROMPT_PATH} 에서 불러와졌습니다.")
        return content  # 파일 내용 반환
    else:
        print(f"파일 {file_path}을(를) 찾을 수 없습니다.\n")
        return None  # 파일이 없으면 None 반환


def PromptResult(RESPONSE, PARAMS, PROMPT_NAME=None, **kwargs):
    """
    대화 결과를 파일에 저장하는 함수.
    RESPONSE가 문자열일 경우 그대로 저장하고,
    리스트나 딕셔너리일 경우 JSON 형식으로 저장.

    PROMPT_PATH: str  # 프롬프트 파일 경로
    PROMPT_NAME: str  # 프롬프트 파일 이름
    PROMPT_EXTENSION: str  # 프롬프트 파일 확장자
    RESULT_PATH: str  # 결과 파일 경로
    RESULT_EXTENSION: str  # 결과 파일 확장자

    PROMPT_NAME이 지정되지 않을 때 PARAMS.PROMPT_NAME을 사용.
    """
    # `prompt_name`이 주어지지 않으면 `PARAMS.PROMPT_NAME`을 사용
    if PROMPT_NAME is None:
        PROMPT_NAME = PARAMS.PROMPT_NAME

    # 현재 년도의 마지막 두 자리 (예: 2024 -> 24)
    year = datetime.now().year % 100

    # 현재 날짜와 시간 (월일시분) 포맷
    timestamp = datetime.now().strftime("%m%d%H%M")

    # RESPONSE가 리스트나 딕셔너리일 경우 RESULT_EXTENSION을 'json'으로 설정
    if isinstance(RESPONSE, (list, dict)):
        PARAMS.RESULT_EXTENSION = "json"  # 리스트나 딕셔너리일 때만 확장자 'json'으로 설정

    # 파일 이름을 PROMPT_NAME, 타임스탬프, 년도의 마지막 두 자리를 결합하여 생성
    file_name = (
        f"{PROMPT_NAME}_result_{year}_{timestamp}.{PARAMS.RESULT_EXTENSION}"
    )

    # 결과 파일 경로 생성
    file_path = os.path.join(PARAMS.RESULT_PATH, file_name)

    # 디렉터리가 존재하지 않으면 생성
    os.makedirs(PARAMS.RESULT_PATH, exist_ok=True)

    # 안내문구 출력
    print(f"결과를 {file_path}에 저장 중입니다...")

    # RESPONSE가 문자열일 경우
    if isinstance(RESPONSE, str):
        # 문자열을 그대로 저장
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(RESPONSE)

    # RESPONSE가 리스트나 딕셔너리일 경우
    elif isinstance(RESPONSE, (list, dict)):
        # 리스트나 딕셔너리를 JSON 형식으로 저장
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(RESPONSE, file, ensure_ascii=False, indent=4)

    else:
        print("지원하지 않는 RESPONSE 형식입니다.")
        return

    print(f"결과가 {file_path}에 {PARAMS.RESULT_EXTENSION} 형식으로 저장되었습니다.\n")


def LLMSupport(QUESTION, PARAMS, SUPPORT_PROMPT="answser question", **kwargs):
    """
    shot 기법 - shot 생성을 위한 함수
    QUESTION : shot 생성을 위한 질문
    SUPPORT_PROMPT : shot 생성을 위한 프롬프트(선택)
    KEY: str  # API Key 환경변수명
    LLM_MODEL: str  # LLM 모델명
    """
    # OpenAI API 키 설정
    openai.api_key = os.environ.get(PARAMS.KEY)
    client = OpenAI(api_key=openai.api_key)

    # ChatCompletion을 사용하여 모델에 요청
    completion = client.chat.completions.create(
        model=PARAMS.LLM_MODEL,
        messages=[
            {"role": "system", "content": SUPPORT_PROMPT},
            {"role": "user", "content": QUESTION},
        ],
        temperature=0.0,
    )

    # 응답에서 답변 내용 추출
    return completion.choices[0].message.content


def PromptTemplate(PARAMS, **kwargs):
    """
    필요 없는 부분은 공백('')으로 작성
    영어 사용, 명령어 사용
    리스트, 마크다운 작성 방식으로 성능을 향상 가능

    PERSONA : LLM이 수행할 역할 지정
    LANG : 답변 생성 언어
    TONE : 답변의 어조 설정
    PURPOSE : 목적 명시
    HOW_WRITE : 답변 방식 예) 개조식
    CONDITION : 추가할 조건
    REFERENCE : 참조
    """
    print(
    'load Prompt Template....'
    )
    # PARAMS 작성 팁 출력
    tip = f"""
    필요 없는 부분은 공백('')으로 작성
    영어 사용, 명령어 사용
    리스트, 마크다운 작성 방식으로 성능을 향상 가능

    PERSONA : LLM이 수행할 역할 지정
    LANG : 답변 생성 언어
    TONE : 답변의 어조 설정
    PURPOSE : 목적 명시
    HOW_WRITE : 답변 방식 예) 개조식
    CONDITION : 추가할 조건
    REFERENCE : 참조
    """
    print(tip)

    # 사용자가 재작성할지 물어보기
    while True:
        user_input = input("현재 PARAMS를 사용하시겠습니까? (y/n): ").strip().lower()

        # 사용자가 'y'를 입력하면 PARAMS 사용
        if user_input == "y":
            break
        # 사용자가 'n'을 입력하면 PARAMS 재작성 안내
        elif user_input == "n":
            print("PARAMS를 다시 작성해주세요.")
            # 사용자가 PARAMS를 수정할 수 있는 로직을 여기에 추가할 수 있습니다.
            break
        else:
            # 다른 키 입력시 다시 물어보기
            print("잘못된 입력입니다. 다시 입력해주세요.")

    # 현재 PARAMS 값으로 포맷된 템플릿 반환
    prompt = f"""
    persona : {PARAMS.PERSONA}
    language : {PARAMS.LANG}
    tone : {PARAMS.TONE}
    purpose : {PARAMS.PERPOSE}
    how to write : {PARAMS.HOW_WRITE}
    condition : {PARAMS.CONDITION}
    reference : {PARAMS.REFERENCE}
    """
    print('Making Prompt... \n')
    print(
        f"""
    persona : {PARAMS.PERSONA}
    language : {PARAMS.LANG}
    tone : {PARAMS.TONE}
    purpose : {PARAMS.PERPOSE}
    how to write : {PARAMS.HOW_WRITE}
    condition : {PARAMS.CONDITION}
    reference : omit
        """
    )

    return prompt
