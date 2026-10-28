from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.generate_description import chat_model


class CollectionText(BaseModel):
    """A photo collection's title and description, rewritten."""

    title: str = Field(
        description=(
            "Short, slightly poetic title (2-6 words). Keep the place names, years and event "
            "names the author used so the collection stays recognisable. No final punctuation."
        )
    )
    description: str = Field(
        description=(
            "One or two short, evocative sentences in the author's voice (first person if they "
            "wrote in first person). Keep every concrete fact they gave: places, dates, events."
        )
    )


# The text is not only read: it is what the collection's photos are searched and matched by, so
# the facts in it (the words a photo's own text would share) have to survive the rewrite.
SYSTEM_PROMPT = (
    "You edit the title and description of a photo collection on a photographer's portfolio. "
    "Rewrite them to read more poetic and evocative. Keep every place, date, year and event the "
    "author mentions, since the collection's photos are found by these words. Do not invent "
    "facts. Write in the same language as the author. Keep it short."
)


class CollectionTextWriter:
    def __init__(self):
        self.model = chat_model.with_structured_output(CollectionText)

    async def rewrite(self, title: str, description: str) -> CollectionText:
        text = f"Title: {title}\nDescription: {description or '(none)'}"
        return await self.model.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=text)]
        )


collection_writer = CollectionTextWriter()
