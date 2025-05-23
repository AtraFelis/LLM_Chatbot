import os
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver


# LangGraph

# state: Node들 간의 데이터를 전달하는 역할
# Node - Node - Node 연결결

class GenLLM:
    def __init__(self):
        _ = load_dotenv(find_dotenv())

        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0
        )
        
        # Graph 정의 (빈 도화지 같은 느낌)
        self.workflow = StateGraph(state_schema=MessagesState)
        
        # 단순 함수
        def call_model(state: MessagesState):
            response = self.llm.invoke(state["messages"])
            return {"messages": response}

        # Graph에 Node 추가
        self.workflow.add_node("model", call_model) # model이라는 이름의 Node 추가
        
        # Graph에 Edge 추가 (방향)
        self.workflow.add_edge(START, "model") # START에서 model로 가는 방향 추가
        
        # workflow [ start -(state)-> model ]
        
        # 메모리 저장
        memory = MemorySaver()
        
        # 사용자의 모든 입력을 메모리에 저장
        self.app = self.workflow.compile(checkpointer=memory)
        
    # self.app 실행행
    def run(self, thread_id:str, user_input:str) -> str:
        human_msg = HumanMessage(content=user_input)
        
        # 사용자 별로 메모리를 구분하기 위해 thread_id 사용 (실제로는 uuid나 domain 등으로 사용)
        # 채팅방 별로 고유한 값이 필요하므로 사용자 아이디 등으로만 구분하면 안됨.
        config = {"configurable": {"thread_id": thread_id}}
        result = self.app.invoke({"messages": [human_msg]}, config=config)
        ai_msg = result["messages"][-1]
        print(f"Assistant: {ai_msg.content}")
        return ai_msg.content
    
gen_llm = GenLLM()

# 사용자의 스레드 아이디
thread = "user_123"

while True:
    q = input("Human: ").strip()
    if not q:
        continue
    gen_llm.run(thread, q)
