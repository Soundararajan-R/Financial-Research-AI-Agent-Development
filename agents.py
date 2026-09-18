import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from apis import get_stock_fundamentals, get_technical_indicators, get_stock_news_sentiment

# Load environment configuration from explicit .env path
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

class FinancialAgent:
    def __init__(self):
        self.tools = [get_stock_fundamentals, get_technical_indicators, get_stock_news_sentiment]
        self.tools_by_name = {t.name: t for t in self.tools}

    def get_llm(self):
        """Reads the key fresh at runtime whenever a query executes."""
        load_dotenv(dotenv_path=env_path, override=True)
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from environment variables.")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024,
            api_key=api_key
        )
        return llm.bind_tools(self.tools)

    def agent_node(self, state: dict) -> dict:
        messages = state["messages"]
        system_instruction = SystemMessage(content="""
You are the Financial Research assistant. 
When introducing yourself, always say: 
"I'm your Financial Research assistant. I can help you explore Indian equities by pulling:"
Then summarize your capabilities (fundamentals, technical indicators like RSI and SMA, and news sentiment) for NSE/BSE stocks.
Always include an educational disclaimer and never provide licensed investment advice.
Do not include the word 'Indian' in your name.
""")
        
        try:
            llm_with_tools = self.get_llm()
            response = llm_with_tools.invoke([system_instruction] + messages)
            return {"messages": [response]}
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "invalid_api_key" in err_str or "Invalid API Key" in err_str or "missing" in err_str:
                error_msg = (
                    "⚠️ **Authentication Error (401 Invalid API Key)**:\n\n"
                    "GROQ_API_KEY is missing or invalid.\n\n"
                    "**Fix:** Ensure your valid Groq API Key is set in the **.env** file."
                )
                print(f"\n[ERROR] GROQ_API_KEY error: {err_str}\n")
                return {"messages": [AIMessage(content=error_msg)]}
            else:
                raise e

    def tool_node(self, state: dict) -> dict:
        messages = state["messages"]
        last_message = messages[-1]
        tool_outputs = []

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for call in last_message.tool_calls:
                name = call["name"]
                args = call["args"]
                tool_fn = self.tools_by_name.get(name)
                result = tool_fn.invoke(args) if tool_fn else f"Error: Tool {name} not found"
                tool_outputs.append(
                    ToolMessage(
                        content=f"{name}_OUTPUT: {str(result)}",
                        tool_call_id=call["id"],
                        name=name
                    )
                )
        return {"messages": tool_outputs}
