import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph, END

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Load a sample text file
loader = TextLoader("sample.txt")
docs = loader.load()

# For demonstration, print a preview
print("Loaded Docs:", docs[:1])


llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)


# Shared state structure
class AgentState(dict): pass

# LLM-based reasoning node
def llm_node(state: AgentState):
    user_question = state["question"]
    context = state["context"]

    prompt = f"""Context:\n{context}\n\nUser Question:\n{user_question}\n\nAnswer:"""
    response = llm.invoke(prompt)
    return AgentState({**state, "llm_answer": response.content})

def human_check_node(state: AgentState):
    answer = state["llm_answer"]
    print("\n🤖 LLM Proposed Answer:")
    print(answer)
    
    choice = input("Do you want to approve this answer? (yes/no): ").strip().lower()
    if choice == "yes":
        return AgentState({**state, "final_answer": answer})
    else:
        correction = input("Please provide the corrected answer:\n")
        return AgentState({**state, "final_answer": correction})

graph = StateGraph(AgentState)

graph.add_node("llm_response", llm_node)
graph.add_node("human_check", human_check_node)

graph.set_entry_point("llm_response")
graph.add_edge("llm_response", "human_check")
graph.add_edge("human_check", END)

runnable = graph.compile()


# Simulate a user question
question = "Summarize this document in one sentence."

# Combine all doc content into context
context = "\n".join([doc.page_content for doc in docs])

state = AgentState({"question": question, "context": context})
result = runnable.invoke(state)

print("\n✅ Final Answer:")
print(result["final_answer"])
