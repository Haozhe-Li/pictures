from langchain_groq import ChatGroq
import dotenv
import os

dotenv.load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0.9
)


async def generate_random_query() -> str:
    messages = [
        (
            "system",
            """You are an imaginative random query generator for a photo gallery. 
                Generate one concise, vivid photo concept (1–5 words). 
                The concepts should vary across diverse themes such as nature, cityscapes, people, architecture, night scenes, seasons, fantasy, and abstract moods. 
                Avoid repetition and clichés. 
                Output only the description, with no extra text or punctuation.

                Example outputs:
                - Misty mountain trail
                - Neon city rain
                - Desert stars
                - Floating lanterns
                - Silent winter lake
            """,
        ),
    ]

    ai_msg = await llm.ainvoke(messages)
    return ai_msg.content
