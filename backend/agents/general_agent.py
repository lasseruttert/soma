from base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate

from ..tools.general_tools import search_tool, wiki_tool, save_tool, chroma_search_tool

class GeneralAgent(BaseAgent):
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             """
             You are a helpful assistant and an expert in the topic you are asked about. 
             Answer the user's question in a concise and informative manner. Use necessary tools.
             """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        tools = [search_tool, wiki_tool, save_tool, chroma_search_tool]
        super().__init__(prompt=prompt, tools=tools)