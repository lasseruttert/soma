from fastapi import FastAPI

from agent.agent import Agent
from vector.vectorstore_db import VectorStoreDB

app = FastAPI()

# endpoint to give the agent a prompt and endpoint to upload documents