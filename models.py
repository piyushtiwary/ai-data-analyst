from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

google_model = init_chat_model(
    model="gemini-2.5-flash",
)

openai_model = model = init_chat_model("gpt-4o-mini")

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