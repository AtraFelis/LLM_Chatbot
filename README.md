### 개발 환경 구축

1. Github Repository 생성
2. 로컬 환경에 Repository 불러오기
3. 가상환경 생성
4. VSCode와 가상환경 연결

### 기존 챗봇 서비스

1. 프롬프트
2. LLM
3. 결과

-> 단방향 구조 : LangChain

### 현재 챗봇

1. 프롬프트
2. LLM
    1. 프롬프트 분석
    2. 프롬프트 확장
    3. 프롬프트 결과
    4. 프롬프트 평가
3. 평가 결과가 부실하면, 다시 2번으로 돌아감

-> 단방향 구조가 아님 : LangGraph

Graph를 구성하는 Node들을 LangChain으로 묶어서 활용

### 프롬프트

``` python
system_prompt = """
    You are a helpful assistant.
    You will be given provided with a set of documents and a question.
"""
```

`You are a helpful assistant.` : 역할 부여
