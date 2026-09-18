import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from explicit .env path
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Set standard output encoding for Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def launch_app():
    """Launches the Streamlit web application."""
    print("Starting Streamlit Web Dashboard...")
    os.system("streamlit run app.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from langchain_core.messages import HumanMessage
        from workflow import financial_graph
        print("\n--- Running CLI Test ---")
        prompt = "Perform a quick technical review for TCS.NS."
        print(f"Query: {prompt}\n")
        try:
            res = financial_graph.invoke({"messages": [HumanMessage(content=prompt)]})
            print(res["messages"][-1].content)
        except Exception as e:
            print(f"CLI Test Note: {e}")
    else:
        launch_app()
