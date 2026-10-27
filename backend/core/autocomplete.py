import os
import re
import string
import asyncio
import time
from typing import Dict, List, Optional, Set, Tuple
from core.trie import Trie
from core.config import settings

# The serialized model lives on the deployment's disk, not in the repo: it is rebuilt from
# Qdrant on every start and whenever images change (see AutocompleteManager).
TRIE_MODEL_PATH = os.getenv("AUTOCOMPLETE_MODEL_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "trie_model.pkl"
)

# After a change, wait this long without further changes before rebuilding, so a batch of
# uploads or edits costs one rebuild. Every new change restarts the wait...
REBUILD_QUIET_SECONDS = 5 * 60
# ...but never postpone the rebuild past this long after the first unbuilt change, so a
# steady trickle of edits cannot starve it.
REBUILD_MAX_WAIT_SECONDS = 30 * 60

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
    """
    Serves suggestions from an in-memory Trie that is rebuilt from Qdrant.

    - `start()` loads the model from disk (so warm restarts answer immediately) and always
      kicks off a fresh rebuild in the background.
    - `schedule_rebuild()` marks the index stale; the rebuild runs after a quiet period.
    - `rebuild_now()` cancels the wait and rebuilds immediately.

    A rebuild builds a brand-new Trie off the event loop and swaps it in at the end, so
    queries always see a complete index. State is per process (one backend instance).
    """

    def __init__(self):
        self.trie = Trie()
        self.is_ready = False
        self.model_path = TRIE_MODEL_PATH
        self.learned_phrases: Set[str] = set()  # Verified bigrams of the current Trie
        self.quiet_seconds = REBUILD_QUIET_SECONDS
        self.max_wait_seconds = REBUILD_MAX_WAIT_SECONDS
        self.is_building = False
        self.last_built_at: Optional[float] = None  # epoch seconds
        self._qdrant = None
        self._build_lock = asyncio.Lock()
        self._wait_task: Optional[asyncio.Task] = None
        self._startup_task: Optional[asyncio.Task] = None
        # Monotonic times of the first/latest change since the last rebuild began; None = nothing pending.
        self._first_change_at: Optional[float] = None
        self._last_change_at: Optional[float] = None

    # --- Lifecycle ---

    def initialize(self):
        """Loads the model saved by a previous run, if there is one."""
        if not os.path.exists(self.model_path):
            print("No saved autocomplete model; it will be built from Qdrant.")
            return
        print(f"Loading autocomplete model from {self.model_path}...")
        try:
            self.trie = Trie.load(self.model_path)
            self.is_ready = True
            print("Autocomplete model loaded successfully.")
        except Exception as e:
            print(f"Failed to load autocomplete model: {e}. It will be rebuilt from Qdrant.")

    def start(self, qdrant_wrapper):
        """Call once Qdrant is reachable: load the saved model, then refresh it in the background."""
        self._qdrant = qdrant_wrapper
        self.initialize()
        self._startup_task = asyncio.create_task(self._rebuild_quietly())

    async def close(self):
        for task in (self._wait_task, self._startup_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # --- Rebuild scheduling ---

    def schedule_rebuild(self):
        """Marks the index stale. Safe to call from any request handler; never blocks."""
        now = time.monotonic()
        if self._first_change_at is None:
            self._first_change_at = now
        self._last_change_at = now
        if self._wait_task is None or self._wait_task.done():
            self._wait_task = asyncio.create_task(self._wait_then_rebuild())

    def _seconds_until_rebuild(self) -> Optional[float]:
        if self._first_change_at is None or self._last_change_at is None:
            return None
        due = min(
            self._last_change_at + self.quiet_seconds,
            self._first_change_at + self.max_wait_seconds,
        )
        return max(0.0, due - time.monotonic())

    async def _wait_then_rebuild(self):
        # Moving `_last_change_at` is what "refreshes the timer": each pass recomputes the
        # deadline, so there is nothing to cancel and restart. Changes that arrive during a
        # rebuild set the timestamps again and are picked up by the next pass.
        while True:
            remaining = self._seconds_until_rebuild()
            if remaining is None:
                return
            if remaining > 0:
                await asyncio.sleep(remaining)
                continue
            self._first_change_at = self._last_change_at = None
            await self._rebuild_quietly()

    async def rebuild_now(self):
        """
        Drops the pending wait and rebuilds immediately, returning once the new index is live.
        If a rebuild is already running, waits for it and then runs another one, because it
        may have read the data before the change that prompted this call. Raises on failure.
        """
        self._first_change_at = self._last_change_at = None
        if self._wait_task and not self._wait_task.done() and not self.is_building:
            self._wait_task.cancel()
        await self._rebuild()

    def status(self) -> Dict:
        return {
            "ready": self.is_ready,
            "building": self.is_building,
            "pending": self._first_change_at is not None,
            "seconds_until_rebuild": self._seconds_until_rebuild(),
            "last_built_at": self.last_built_at,
        }

    # --- Rebuild ---

    async def _rebuild_quietly(self):
        """
        Background rebuild. A failure is logged and the current index keeps serving; the
        rebuild is scheduled again so a transient outage (e.g. Qdrant) heals by itself.
        """
        try:
            await self._rebuild()
        except Exception as e:
            print(f"Autocomplete rebuild failed, will retry: {e}")
            self.schedule_rebuild()

    async def _rebuild(self):
        async with self._build_lock:
            self.is_building = True
            try:
                print("Rebuilding autocomplete index from Qdrant...")
                texts = await self._fetch_texts()
                trie, phrases = await asyncio.to_thread(self._build_trie, texts)
                # Swap in the finished index in one step.
                self.trie, self.learned_phrases = trie, phrases
                self.is_ready = True
                self.last_built_at = time.time()
                print(f"Autocomplete index rebuilt from {len(texts)} text segments.")
                try:
                    await asyncio.to_thread(trie.save, self.model_path)
                except Exception as e:
                    print(f"Failed to save autocomplete model to {self.model_path}: {e}")
            finally:
                self.is_building = False

    async def _fetch_texts(self) -> List[str]:
        """Every title and description in the collection."""
        texts: List[str] = []
        offset = None
        while True:
            records, offset = await self._qdrant.client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for point in records:
                payload = point.payload or {}
                if payload.get("title"):
                    texts.append(payload["title"])
                if payload.get("description"):
                    texts.append(payload["description"])
            if not records or offset is None:
                return texts

    def _build_trie(self, texts: List[str]) -> Tuple[Trie, Set[str]]:
        """CPU-bound: learns common phrases, then inserts every text. Run off the event loop."""
        # Pre-tokenize exactly as insertion will, so the learned bigrams match.
        detector = SimplePhraseDetector(stop_words=STOP_WORDS)
        detector.learn_vocab([s for text in texts for s in self._tokenize_to_sentences(text)])
        phrases = detector.get_phrases()

        trie = Trie()
        for text in texts:
            self._process_and_insert(text, trie, phrases)
        return trie, phrases

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

    def _process_and_insert(self, text: str, trie: Trie, learned_phrases: Set[str]):
        # reuse tokenize helper
        sentences = self._tokenize_to_sentences(text)

        for words in sentences:
            # 1. Apply Phrase Merging based on the learned phrases
            if learned_phrases:
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
                        if bg in learned_phrases:
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
                
                trie.insert(word, phrase=suffix_phrase)

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
