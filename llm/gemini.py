import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from .rate_limiter import RateLimiter

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Copy .env.example to .env and set it.")

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.1-flash")
VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", CHAT_MODEL)
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

rate_limiter = RateLimiter(
    float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL", "4"))
)

vision_rate_limiter = RateLimiter(
    float(os.getenv("GEMINI_VISION_MIN_REQUEST_INTERVAL", "1.5"))
)

chat_model = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    google_api_key=API_KEY,
    temperature=0,
)

vision_model = ChatGoogleGenerativeAI(
    model=VISION_MODEL,
    google_api_key=API_KEY,
    temperature=0,
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=API_KEY,
)

def extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content or "")

def invoke_chat(messages):
    response = rate_limiter.run(lambda: chat_model.invoke(messages))
    response.content = extract_text_from_content(response.content)
    return response

def invoke_vision(messages):
    response = vision_rate_limiter.run(lambda: vision_model.invoke(messages))
    response.content = extract_text_from_content(response.content)
    return response

def embed_documents(texts):
    # One LangChain batch call rather than one request per chunk.
    return rate_limiter.run(lambda: embedding_model.embed_documents(texts))

def embed_query(text):
    return rate_limiter.run(lambda: embedding_model.embed_query(text))

