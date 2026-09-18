import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment configuration
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

def test_connection():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    print(f"Testing GROQ_API_KEY connectivity (Key starts with: '{api_key[:7]}...', length: {len(api_key)})...")
    
    if not api_key or api_key in ["your_groq_api_key_here", "placeholder_key"]:
        print("[FAIL] GROQ_API_KEY is missing or invalid.")
        return False

    try:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_tokens=50,
            api_key=api_key
        )
        response = llm.invoke("Say 'HTTP 200 Connection Successful' in one short sentence.")
        print(f"[SUCCESS] Groq API Response: {response.content}")
        return True
    except Exception as e:
        print(f"[FAIL] Connectivity Error: {e}")
        return False

if __name__ == "__main__":
    test_connection()
