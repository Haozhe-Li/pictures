import asyncio
import json
import re
from typing import Dict

from groq import Groq


class ImageDescriptionGenerator:
    """
    Generate a short title and a detailed description for an image.
    """

    def __init__(self):
        self.client = Groq()
        self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"

    async def generate(self, image_base64: str) -> Dict[str, str]:
        return await asyncio.to_thread(self._generate_sync, image_base64)

    def _generate_sync(self, image_base64: str) -> Dict[str, str]:
        image_url = f"data:image/jpeg;base64,{image_base64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            """You are an image understanding assistant. Generate a poetic title and a brief description for the provided image.  

Requirements:  
1) Output **JSON only** in the format: {"title":"...","description":"..."}  
2) The **title** should be **very short (2–6 words)**, slightly poetic, and not a literal description of the image. **No punctuation.**  
3) The **description** should be **one short, poetic sentence**, ending with a location tag in the format “City, Country” (e.g., “Dreamlike mist over quiet hills. Kyoto, Japan”).  
4) If the location is in the United States, use “City, State” (e.g., “Golden fields beneath the sun. Yosemite, CA”).  
5) If the image features a famous or recognizable landmark or district, include that in the location tag, formatted as “Landmark, City, Country” (e.g., “Shibuya, Tokyo, Japan”).  
6) Both title and description must be in **English**.  
7) No Markdown, no explanations, and no extra fields — only return the JSON output.  

"""
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            max_completion_tokens=512,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or ""
        return _parse_llm_json(content)


def _parse_llm_json(text: str) -> Dict[str, str]:
    data = {}
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = {}

    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()

    if not title and not description:
        return {"title": "", "description": text.strip()}

    return {"title": title, "description": description}


description_generator = ImageDescriptionGenerator()
