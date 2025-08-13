from langgraph.graph import StateGraph, START, END

from ..utils.state import State
from ..agents.general_agent import GeneralAgent

general_agent = GeneralAgent()

def classify_message(state: State) -> State:
    # Implement your classification logic here
    return state


graph_builder = StateGraph(State)

graph_builder.add_node("general", general_agent)

graph_builder.add_node("classify", classify_message)

graph_builder.add_edge(START, "classify")

graph_builder.add_edge("classify", "general")

graph_builder.add_edge("general", END)

app = graph_builder.compile()

if __name__ == "__main__":
    state = {"messages": [], "category": "general"}
    while True: 
        input_text = input("Enter your message: ")
        if input_text.lower() == "exit":
            break
        state["messages"].append({"role": "user", "content": input_text})
        state = app.invoke(state)
        last_message = state.get("messages", [])[-1]
        print("Agent:", last_message.get("content", ""))