from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# google_model = init_chat_model(
#     model="gemini-2.5-flash",
# )


openai_model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

if __name__ == "__main__":

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = openai_model.invoke(messages)
    print(ai_msg)
