
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate


_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

system_prompt = """
    당신은 소설을 쓰는 작가를 보조하는 역할입니다.
    전달받은 글을 다음 조건에 맞게 한국어에서 '{lang}'으로 번역해주세요.
    
    조건1. 작가의 질문에 절대로 답변하지마세요.
    조건2. 작가의 글을 문자열 그대로 번역해주세요.
    조건3. 문법에 신경서서 번역해주세요.
    조건4. 줄임말과 신조어 등은 최대한 사용하지마세요.
    조건5. 판타지 소설에 많이 사용하는 문체와 어조로 번역해주세요.
    조건6. 번역시 기존 글의 스타일을 유지해주세요.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{novel}")
])

novel = """
    무한 회귀.
    그런 이름의 장르가 있다.

    주인공이 죽음을 겪으면 다시 죽기 이전의 상태로 되돌아가서 끊임없이 난관에 도전하는 것을 ‘무한 회귀’라 일컫는다.

    당연하지만 주인공은 아무리 험난한 난관이 닥쳐도 어떻게든 극복해 낸다. 그야 극복할 때까지 도전하면 그만이니까.

    본래는 배드 엔딩으로 끝나야 할 운명이 해피 엔딩으로 바뀐다든지, 불치병에 걸려 죽어야 하는 서브 히로인을 주인공이 기적적으로 구한다든지――.
    무한 회귀란 모든 비극을 종결시키는 치트키나 다름없다.

    하지만 직접 겪어 본 경험자로서 말하건대, 각종 소설에서 묘사된 무한 회귀는 저열한 프로파간다에 불과하다.
    마치 명문대에 합격한 학생들의 이름만 현수막으로 내거는 입시학원처럼 말이다.
"""

lang = "영어"

novel_input = {
    "novel": novel,
    "lang": lang
}


chain = prompt | llm
result = chain.invoke(novel_input)
print(result.content)

