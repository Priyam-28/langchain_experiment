import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from typing import Dict, List
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import tool
import json

load_dotenv()

# Set up API keys
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

class NotApproved(Exception):
    """Custom exception."""

# Adding human approval
def human_approval(msg: AIMessage) -> AIMessage:
    """Responsible for passing through its input or raising an exception.

    Args:
        msg: output from the chat model

    Returns:
        msg: original output from the msg
    """
    tool_strs = "\n\n".join(
        json.dumps(tool_call, indent=2) for tool_call in msg.tool_calls
    )
    input_msg = (
        f"Do you approve of the following tool invocations\n\n{tool_strs}\n\n"
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.\n >>>"
    )
    resp = input(input_msg)
    if resp.lower() not in ("yes", "y"):
        raise NotApproved(f"Tool invocations not approved:\n\n{tool_strs}")
    return msg


@tool
def count_emails(last_n_days: int) -> int:
    """Dummy function to count number of e-mails. Returns 2 * last_n_days."""
    return last_n_days * 2


@tool
def send_email(message: str, recipient: str) -> str:
    """Dummy function for sending an e-mail."""
    return f"Successfully sent email to {recipient}."


tools = [count_emails, send_email]
llm_with_tools = llm.bind_tools(tools)


def call_tools(msg: AIMessage) -> List[Dict]:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


chain = llm_with_tools | human_approval | call_tools
ans=chain.invoke("how many emails did i get in the last 5 days?")
print(ans)