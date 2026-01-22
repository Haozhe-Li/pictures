from langchain_groq import ChatGroq
import dotenv
import os

dotenv.load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0.9)


async def generate_random_query() -> str:
    messages = [
        (
            "system",
            """Generate one concise, vivid photo description (<5 words). Output only the description with no extra text.
Example: Star gazing""",
        ),
    ]
    ai_msg = await llm.ainvoke(messages)
    return ai_msg.content
