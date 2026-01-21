import os
import string
import asyncio
from typing import List
from core.trie import Trie
from core.config import settings

# Define the path for the serialized model
TRIE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trie_model.pkl")

class AutocompleteManager:
    def __init__(self):
        self.trie = Trie()
        self.is_ready = False
        self.model_path = TRIE_MODEL_PATH

    def initialize(self):
        """Attempts to load the model from disk on startup."""
        if os.path.exists(self.model_path):
            print(f"Loading autocomplete model from {self.model_path}...")
            try:
                self.trie = Trie.load(self.model_path)
                self.is_ready = True
                print("Autocomplete model loaded successfully.")
            except Exception as e:
                print(f"Failed to load autocomplete model: {e}. Starting with empty Trie.")
        else:
            print("No autocomplete model found. Please trigger build via /api/autocomplete/build.")

    async def build_index(self, qdrant_wrapper):
        """Fetches data from Qdrant and rebuilds the Trie."""
        print("Rebuilding autocomplete index from Qdrant...")
        self.trie = Trie() # Reset
        
        offset = None
        count = 0
        total_points = 0
        
        print(f"Using collection: {settings.COLLECTION_NAME}")

        while True:
            records, next_offset = await qdrant_wrapper.client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
            
            if not records:
                break
                
            for point in records:
                payload = point.payload
                if not payload: continue

                # We primarily want to autocomplete Titles as full phrases.
                if payload.get("title"):
                    self._process_and_insert(payload["title"])
                
                # For descriptions, we also want to index them
                if payload.get("description"):
                    self._process_and_insert(payload["description"])
            
            count += len(records)
            offset = next_offset
            if offset is None:
                break
        
        # Save the built model
        try:
            self.trie.save(self.model_path)
            self.is_ready = True
            print(f"Autocomplete index built with {count} records and saved to {self.model_path}.")
        except Exception as e:
            print(f"Failed to save Trie model: {e}")

    def _process_and_insert(self, text: str, is_phrase: bool = False):
        # Basic tokenization
        # Remove punctuation and split by whitespace
        # We want to keep the original casing for the stored phrase if possible, 
        # but for simplicity in this Trie which lowercases everything in `insert`,
        # we will just store the cleaned lowercased suffix.
        # Ideally, we should store the original text slice, but let's stick to lower for consistency.
        
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        clean_text = text.translate(translator).lower()
        words = clean_text.split()
        
        # Generate suffixes
        # "Osaka Castle Park" -> ["osaka castle park", "castle park", "park"]
        for i in range(len(words)):
            word = words[i]
            if len(word) < 2: continue # Skip single chars
            
            # Construct suffix from this word onwards
            # Limit suffix length to e.g. 4 words (Matched word + 3 following)
            suffix_words = words[i : i+4] 
            suffix_phrase = " ".join(suffix_words)
            
            self.trie.insert(word, phrase=suffix_phrase)

    def suggest(self, query: str) -> List[str]:
        if not query or not self.is_ready:
            return []
        
        parts = query.split()
        if not parts:
            return []

        # Restriction 1: If query has more than 5 words, do not autocomplete
        if len(parts) > 5:
            return []
        
        # Determine strictness based on user input
        if query.endswith(" "):
            # User typed a space, usually expects next word, but our Trie is word-based.
            # We don't have n-gram or next-word prediction here yet.
            # So we return nothing or maybe everything? Returning nothing is safer.
            return []
        
        # We complete the last word currently being typed
        last_word = parts[-1]
        
        # Fuzzy search on the last word
        # max_distance=1 allows for 1 typo
        suggestions = self.trie.search_fuzzy(last_word, max_distance=1, limit=5)
        
        # If the search returns phrases (which contain spaces), we directly return them
        # as they are usually better suggestions than reconstructing words.
        # But if the suggestion is just a single word (from description), we might still want to reconstruct.
        # FIX: Always prepend prefix to support "breakpoint completion" context properly.
        # e.g., Query: "man standing in yosemite" -> Prefix: "man standing in", Suggestion: "yosemite valley"
        # Result should be: "man standing in yosemite valley"
        
        final_results = []
        prefix = " ".join(parts[:-1])

        for s in suggestions:
            # Restriction 2: Truncate suggestion to max 4 words (Matched + 3 following)
            # This handles cases where older models might have longer phrases stored
            s_parts = s.split()
            if len(s_parts) > 4:
                s = " ".join(s_parts[:4])
            
            if prefix:
                final_results.append(f"{prefix} {s}")
            else:
                final_results.append(s)
        
        # Deduplicate and limit
        return list(set(final_results))[:10]

autocomplete_manager = AutocompleteManager()
