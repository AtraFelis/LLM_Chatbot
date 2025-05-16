from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
from langchain.prompts import ChatPromptTemplate

# 1. env 불러오기
# find_dotenv() 환경변수 파일 경로 찾기
# load_dotenv() 환경변수 로드
_ = load_dotenv(find_dotenv())

# 2. llm 모델 생성
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    # temperature 0 : 예측 가능한 답변
    # temperature 1 : 다양한 답변
    temperature=0
)

# 3. 프롬프트 생성
system_prompt = """
    You are a helpful assistant.
    You will be given provided with a set of documents and a question.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# 4. Chain 생성
chain = prompt | llm

# 5. Chain 실행
question = "내 이름은 송채선이야"
result = chain.invoke({"question": question})
print(result)