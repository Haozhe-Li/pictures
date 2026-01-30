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

# Words that should remain lowercase in title case (unless they are the first word)
NON_CAPITALIZED_WORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "for", 
    "with", "by", "from", "and", "or", "as", "but", "nor"
}

class SimplePhraseDetector:
    """
    A lightweight statistical bigram detector to merge common phrases like 'San Francisco'.
    Inspired by Gensim's Phrases model but simplified to avoid dependencies.
    """
    def __init__(self, min_count=3, threshold=10.0, stop_words=None):
        self.min_count = min_count
        self.threshold = threshold
        self.vocab = {}
        self.bigram_counts = {}
        self.stop_words = stop_words or set()

    def learn_vocab(self, sent_tokens_list: List[List[str]]):
        """Pass 1: Count unigrams and bigrams."""
        for tokens in sent_tokens_list:
            for i, token in enumerate(tokens):
                self.vocab[token] = self.vocab.get(token, 0) + 1
                
                if i < len(tokens) - 1:
                    w1, w2 = token, tokens[i+1]
                    # Don't form phrases starting/ending with typically 'connector' stop words 
                    # unless dealing with very specific nouns. 
                    # For now, let's be permissive but rely on the statistical threshold to filter junk.
                    # We skip bigrams where BOTH are stop words (e.g. "in the", "to be")
                    if w1 in self.stop_words and w2 in self.stop_words:
                         continue

                    bg = f"{w1} {w2}"
                    self.bigram_counts[bg] = self.bigram_counts.get(bg, 0) + 1

    def get_phrases(self):
        """Pass 2: Identify phrases based on score."""
        phrases = set()
        vocab_len = len(self.vocab)
        
        for bg, count in self.bigram_counts.items():
            if count < self.min_count:
                continue
            
            w1, w2 = bg.split(" ")
            count1 = self.vocab.get(w1, 0)
            count2 = self.vocab.get(w2, 0)
            
            if count1 == 0 or count2 == 0: continue
            
            # Score formula: (bigram_count - min) * N / (count1 * count2)
            score = (count - self.min_count) * vocab_len / (count1 * count2)
            
            if score > self.threshold:
                phrases.add(bg)
        
        print(f"Learned {len(phrases)} common phrases (e.g. {list(phrases)[:5] if phrases else 'None'})")
        return phrases

class AutocompleteManager:
    def __init__(self):
        self.trie = Trie()
        self.is_ready = False
        self.model_path = TRIE_MODEL_PATH
        self.learned_phrases = set() # Store verified bigrams

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
        self.learned_phrases = set() # Reset
        
        offset = None
        count = 0
        total_points = 0
        
        print(f"Using collection: {settings.COLLECTION_NAME}")

        # Phase 1: Collect all text to learn phrases
        all_texts = [] 
        
        # We need to scroll through everything first anyway.
        # Ideally we loop once to collect text, train, then insert.
        # Note: If dataset is massive (millions), saving all_texts in RAM is bad.
        # But for typical gallery ( < 100k), 100k strings is fine (~20MB).
        
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

                if payload.get("title"):
                    all_texts.append(payload["title"])
                
                if payload.get("description"):
                    all_texts.append(payload["description"])
            
            count += len(records)
            offset = next_offset
            if offset is None:
                break
        
        print(f"Collected {len(all_texts)} text segments. Learning phrases...")
        
        # Train Bigram Detector
        detector = SimplePhraseDetector(stop_words=STOP_WORDS)
        # We need to pre-tokenize exactly as we do during insertion
        # We'll use a helper _tokenize(text) -> list[list[str]] (sentences -> words)
        tokenized_corpus = []
        for text in all_texts:
            sentences = self._tokenize_to_sentences(text)
            tokenized_corpus.extend(sentences)
            
        detector.learn_vocab(tokenized_corpus)
        self.learned_phrases = detector.get_phrases()
        
        # Phase 2: Build Trie using learned phrases
        print("Inserting into Trie...")
        for text in all_texts:
            self._process_and_insert(text)
        
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

    def _tokenize_to_sentences(self, text: str) -> List[List[str]]:
        """
        Helper to consistently tokenize text into sentences of words.
        Returns: [ ["word1", "word2"], ["word3", "word4"] ]
        """
        segments = re.split(r'[.!?;:\n]+', text)
        punc_to_remove = string.punctuation.replace("'", "") 
        translator = str.maketrans(punc_to_remove, ' ' * len(punc_to_remove))

        result = []
        for segment in segments:
            if not segment.strip(): continue
            # clean_text = segment.translate(translator).lower() # Keep lower for consistency
            clean_text = segment.translate(translator).lower()
            words = clean_text.split()
            if words:
                result.append(words)
        return result

    def _process_and_insert(self, text: str, is_phrase: bool = False):
        # reuse tokenize helper
        sentences = self._tokenize_to_sentences(text)

        for words in sentences:
            # 1. Apply Phrase Merging based on self.learned_phrases
            if self.learned_phrases:
                merged_words = []
                skip_next = False
                for k in range(len(words)):
                    if skip_next:
                        skip_next = False
                        continue
                    
                    word = words[k]
                    # Check forward bigram
                    if k < len(words) - 1:
                        next_word = words[k+1]
                        bg = f"{word} {next_word}"
                        if bg in self.learned_phrases:
                            merged_words.append(bg)
                            skip_next = True
                            continue
                    
                    merged_words.append(word)
                words = merged_words

            # 2. Insert into Trie
            for i in range(len(words)):
                word = words[i]
                if len(word) < 2 and word not in ("a", "i"): continue 
                
                # Construct suffix using smart limit 
                # Note: words list now contains merged phrases like "san francisco"
                suffix_phrase = self._get_smart_suffix(words, i, limit=3)
                
                self.trie.insert(word, phrase=suffix_phrase)

    def _smart_title_case(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        
        capitalized_words = []
        for i, word in enumerate(words):
            # First word is always capitalized
            if i == 0:
                capitalized_words.append(word.capitalize())
            # For subsequent words, check if they are in the non-capitalized list
            elif word.lower() in NON_CAPITALIZED_WORDS:
                capitalized_words.append(word.lower())
            else:
                capitalized_words.append(word.capitalize())
        
        return " ".join(capitalized_words)

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
                full_phrase = f"{prefix} {truncated_s}"
            else:
                full_phrase = truncated_s
            
            final_results.append(self._smart_title_case(full_phrase))
        
        # Deduplicate and limit
        return list(set(final_results))[:10]

autocomplete_manager = AutocompleteManager()
