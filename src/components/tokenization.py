"""Tokenization: text → token IDs (word-level with special tokens)."""
import numpy as np
from typing import List, Dict


class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        special_tokens = [self.pad_token, self.unk_token , self.bos_token, self.eos_token]
        for id, token in enumerate(special_tokens):
            self.word_to_id[token] = id
        words = []
        for text in texts:
            word = text.split()
            words.extend(word)
        words_lower = [word.lower() for word in words]
        unique_words = sorted(set(words_lower))

        for id, token in enumerate(unique_words):
            start = len(special_tokens)
            self.word_to_id[token] = id+start
        self.vocab_size = len(unique_words) + len(special_tokens)
        for id, sp_tokens in enumerate(special_tokens):
            self.id_to_word[id] = sp_tokens
        for id, tokens in enumerate(unique_words):
            self.id_to_word[id+len(special_tokens)] = tokens

            
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        list_id = [self.word_to_id.get(word, self.word_to_id[self.unk_token]) for word in words]
        return list_id
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        list_text = [self.id_to_word.get(id, self.unk_token) for id in ids]
        return " ".join(list_text)