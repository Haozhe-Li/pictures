import pickle
import os
from typing import List, Set

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word = None
        self.phrases = set()  # Store full phrases that contain this word

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, phrase: str = None):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.word = word
        if phrase:
            node.phrases.add(phrase)

    def search(self, prefix: str, limit: int = 10) -> List[str]:
        """Exact prefix search returning associated phrases."""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        
        results = set()
        self._dfs(node, results, limit)
        return list(results)

    def _dfs(self, node: TrieNode, results: Set[str], limit: int):
        if len(results) >= limit:
            return
        
        if node.is_end_of_word:
            # If we have phrases, prefer them. If not, fallback to the word itself.
            if node.phrases:
                for p in node.phrases:
                    if len(results) >= limit: break
                    results.add(p)
            else:
                results.add(node.word)
        
        for char in node.children:
            self._dfs(node.children[char], results, limit)

    def search_fuzzy(self, pattern: str, max_distance: int = 1, limit: int = 10) -> List[str]:
        """Fuzzy search allowing insertions, deletions, and substitutions."""
        results = set()
        # Start DFS from root
        self._dfs_fuzzy(self.root, pattern.lower(), 0, max_distance, results, limit)
        return list(results)

    def _dfs_fuzzy(self, node: TrieNode, pattern: str, index: int, edits_left: int, results: Set[str], limit: int):
        if len(results) >= limit:
            return

        # If we have reached the end of the pattern, we collect words from this node downwards
        # This treats the pattern as a prefix (with potential errors)
        if index == len(pattern):
             self._collect_all(node, results, limit)
             return

        char = pattern[index]

        # 1. Match: Pattern matches Trie path
        if char in node.children:
            self._dfs_fuzzy(node.children[char], pattern, index + 1, edits_left, results, limit)
        
        if edits_left > 0:
            # 2. Substitution: Pattern matches Trie path with a change
            for child_char in node.children:
                self._dfs_fuzzy(node.children[child_char], pattern, index + 1, edits_left - 1, results, limit)
            
            # 3. Insertion: Pattern missed a char that is in Trie (Trie advances, Pattern stays)
            for child_char in node.children:
                self._dfs_fuzzy(node.children[child_char], pattern, index, edits_left - 1, results, limit)

            # 4. Deletion: Pattern has extra char not in Trie (Trie stays, Pattern advances)
            self._dfs_fuzzy(node, pattern, index + 1, edits_left - 1, results, limit)

    def _collect_all(self, node: TrieNode, results: Set[str], limit: int):
        if len(results) >= limit:
            return
        if node.is_end_of_word:
            # If we have phrases, prefer them. If not, fallback to the word itself.
            if node.phrases:
                for p in node.phrases:
                    if len(results) >= limit: break
                    results.add(p)
            else:
                results.add(node.word)
        for char in node.children:
            self._collect_all(node.children[char], results, limit)

    def save(self, filepath: str):
        """Serialize the Trie to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'Trie':
        """Load the Trie from a file."""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading Trie: {e}")
            return cls()
