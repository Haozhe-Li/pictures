import os
import re
import string
import asyncio
from typing import List
from core.trie import Trie
from core.config import settings

# Define the path for the serialized model
TRIE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trie_model.pkl")

# Define a minimal set of stop words for autocomplete logic
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "for", 
    "with", "by", "from", "and", "or", "is", "are", "was", "were",
    "it", "that", "this", "my", "your", "as", "but", "be"
}

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

    def _get_smart_suffix(self, words: list, start_index: int, limit: int = 3) -> str:
        """
        Extracts a suffix starting from start_index.
        Includes at most `limit` significant words (non-stop-words) after the first word.
        """
        result = [words[start_index]] # Always keep the matched word
        significant_count = 0
        idx = start_index + 1
        
        while idx < len(words) and significant_count < limit:
            word = words[idx]
            result.append(word)
            
            if word not in STOP_WORDS:
                significant_count += 1
            
            idx += 1
            
            # Hard limit to prevent runaway strings if everything is a stop word
            if len(result) > 10: 
                break
        
        # Optional: Trim trailing stop words if we hit the limit?
        # E.g. "walking in the park and" -> "walking in the park"
        while len(result) > 1 and result[-1] in STOP_WORDS:
            result.pop()
            
        return " ".join(result)

    def _process_and_insert(self, text: str, is_phrase: bool = False):
        # Improved logic:
        # 1. Split by sentence terminators/punctuation to avoid crossing semantic boundaries.
        #    e.g. "xxxx. SF, CA" -> ["xxxx", "SF, CA"]
        #    This prevents "xxxx" from suggesting "SF, CA".
        segments = re.split(r'[.!?;:\n]+', text)

        # Prepare punctuation translator:
        # We replace most punctuation with space, but we might want to handle apostrophes differently 
        # (e.g., keep them attached to words like "don't", "O'Connor").
        # For now, let's keep the original logic mapping all punctuation to space for simplicity/consistency,
        # unless specifically requested to handle apostrophes effectively. 
        # User asked for "Better" behavior - preserving intra-word apostrophes is generally better.
        
        punc_to_remove = string.punctuation.replace("'", "") # Keep '
        translator = str.maketrans(punc_to_remove, ' ' * len(punc_to_remove))

        for segment in segments:
            if not segment.strip(): continue

            # Clean segment: keep ' but lower case, remove other punctuation
            clean_text = segment.translate(translator).lower()
            words = clean_text.split()
            
            # Generate suffixes for this segment only
            for i in range(len(words)):
                word = words[i]
                # Filter out pure punctuation remnants if any (though translate handles most)
                if len(word) < 2 and word not in ("a", "i"): continue 
                
                # Construct suffix using smart limit (limit=3 significant words)
                suffix_phrase = self._get_smart_suffix(words, i, limit=3)
                
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
            # Restriction 2: Re-apply smart truncation in case older models are loaded
            # or data inconsistency.
            s_parts = s.split()
            # If the stored phrase is longer than we expect (though we trimmed at insert time),
            # we can run the smart truncation again effectively.
            # Reuse logic: treat the whole phrase as "words" starting at 0
            truncated_s = self._get_smart_suffix(s_parts, 0, limit=3)
            
            if prefix:
                final_results.append(f"{prefix} {truncated_s}")
            else:
                final_results.append(truncated_s)
        
        # Deduplicate and limit
        return list(set(final_results))[:10]

autocomplete_manager = AutocompleteManager()
