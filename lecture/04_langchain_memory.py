import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory
from langchain.schema.runnable import RunnablePassthrough

class GenLLM:
    def __init__(self):
        _ = load_dotenv(find_dotenv())

        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0
        )
        
        # 대화기록을 저장할 Memory
        self.memory = ConversationSummaryMemory( # 대화 기록을 요약하여 저장하는 메모리
            llm = self.llm, # 대화 기록을 요약할 때 필요한 모델
            max_token_limit=50,
            memory_key="chat_history",
            return_messages=True
        )
        
        self.system_prompt = """
            You are a helpful assistant.
            You will be provided with a set of documents and a question.
        """
    
    # Memory에 저장된 대화기록 불러오기
    def load_memory(self, input):
        print(input)
        return self.memory.load_memory_variables({})["chat_history"]
    
    def gen_prompt(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # RunnablePassthrough.assign() : 입력 변수를 그대로 전달 -> 현재는 메모리에 저장된 대화기록을 불러와서 전달함
        chain = RunnablePassthrough.assign(chat_history=self.load_memory) | prompt | self.llm
        return chain

    def run_chain(self, question:str):
        chain = self.gen_prompt()
        result = chain.invoke({"question": question})
        self.memory.save_context(
            {"input": question},
            {"output": result.content}
        )
        print(f"Assistant: {result.content}")
        print("Memory:", self.memory.load_memory_variables({})["chat_history"])
        return result.content

gen_llm = GenLLM()

while True:
    gen_llm.run_chain(input("Human: "))