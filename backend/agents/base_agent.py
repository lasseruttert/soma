import os
from dotenv import load_dotenv

from ..utils.state import State

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor


load_dotenv()

class BaseAgent:
    def __init__(self, prompt = None, tools = None):
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL"),
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.prompt = prompt if prompt is not None else ChatPromptTemplate.from_messages([
            ("system", 
             """
             You are a helpful assistant and an expert in the topic you are asked about. 
             Answer the user's question in a concise and informative manner. Use necessary tools.
             """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        self.tools = tools if tools is not None else []
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=self.tools
        )
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False)
        
    def __call__(self, state : State):
        user_message = state.get("messages", [{}])[-1].get("content", "")
        response = self.executor.invoke({"input": user_message, "chat_history": state.get("messages", [])})
        state["messages"].append({"role": "assistant", "content": response.get("output", response)})
        return state