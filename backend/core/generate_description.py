from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Shared by every LLM call in the backend (captions here, query rewriting in random_query).
# Reasoning is switched off; everything else is left at the defaults.
chat_model = ChatOpenAI(model="gpt-5.6-luna", reasoning_effort="none")


class ImageDescription(BaseModel):
    """A title and a short caption for a photograph."""

    title: str = Field(
        description=(
            "Very short (2-6 words), slightly poetic title that is not a literal "
            "description of the image. English, no punctuation."
        )
    )
    description: str = Field(
        description=(
            "One short, poetic English sentence that ends with a location tag in the "
            'format "City, Country" (e.g. "Dreamlike mist over quiet hills. Kyoto, Japan"). '
            'In the United States use "City, State" (e.g. "Golden fields beneath the sun. '
            'Yosemite, CA"). For a famous or recognizable landmark or district use '
            '"Landmark, City, Country" (e.g. "Shibuya, Tokyo, Japan").'
        )
    )


PROMPT = "You are an image understanding assistant. Generate a poetic title and a brief description for the provided image."


class ImageDescriptionGenerator:
    """
    Generate a short title and a detailed description for an image.
    """

    def __init__(self):
        self.model = chat_model.with_structured_output(ImageDescription)

    async def generate(self, image_base64: str) -> ImageDescription:
        message = HumanMessage(
            content=[
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ]
        )
        return await self.model.ainvoke([message])


description_generator = ImageDescriptionGenerator()
