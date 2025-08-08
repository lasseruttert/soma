from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from backend.tools.tools import search_tool, wiki_tool, save_tool, chroma_search_tool

load_dotenv()

class ResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools: list[str]
    

import os

class Agent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL"),
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.output_parser = PydanticOutputParser(pydantic_object=ResponseModel)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             """
             You are a helpful assistant and an expert in the topic you are asked about. 
             Answer the user's question in a concise and informative manner. Use necessary tools.
             Wrap the output in this format: {format_instructions}
             """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        self.tools = [search_tool, wiki_tool, save_tool, chroma_search_tool]
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=self.tools
        )
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, input_text: str):
        raw_response = self.executor.invoke({"input": input_text})
        try:
            response = self.output_parser.parse(raw_response.get("output", ""))
            return response
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None
        
if __name__ == "__main__":
    agent = Agent()
    response = agent.run("Who is John Doe? Use the Chroma Search Tool")
    print(response)