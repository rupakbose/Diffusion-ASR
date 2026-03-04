import re
import json
import torch
from pathlib import Path
from collections import Counter

class WordTokenizer:
    """Tokenizer for ASR supporting both Word-level and Character-level."""
    
    def __init__(self, level="word", pad_token="<PAD>", unk_token="<UNK>", bos_token="<BOS>", eos_token="<EOS>", mask_token="<MASK>"):
        self.level = level.lower() # "word" or "char"
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token, mask_token]
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def _clean_text(self, text):
        """Removes punctuation and converts to uppercase. Keeps spaces."""
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.upper()
        
        if self.level == "word":
            return text.strip() 
        return text 

    def _tokenize(self, clean_text):
        """Internal helper to split text. Preserves special tokens in char level."""
        if self.level == "word":
            return clean_text.split()
        
        
        special_pattern = "|".join(map(re.escape, self.special_tokens))
        tokens = re.findall(f"{special_pattern}|.", clean_text)
        
        return tokens

    def fit(self, transcript_files, min_freq=1):
        """Builds vocabulary based on word or character frequency."""
        counter = Counter()
        lengths = []
        
        for file_path in transcript_files:
            with open(file_path, 'r') as f:
                for line in f:
                    text = line.strip().split(" ", 1)[-1]
                    clean_text = self._clean_text(text)
                    units = self._tokenize(clean_text)
                    
                    counter.update(units)
                    lengths.append(len(units) + 2)

       
        self.word2idx = {token: i for i, token in enumerate(self.special_tokens)}
        
      
        sorted_units = sorted(counter.items()) if self.level == "char" else counter.items()
        
        for unit, count in sorted_units:
            if count >= min_freq and unit not in self.word2idx:
                self.word2idx[unit] = len(self.word2idx)
        
        self._update_reverse_vocab()

        if lengths:
            print("\n" + "="*30)
            print(f"CORPUS STATISTICS ({self.level.upper()} LEVEL)")
            print("="*30)
            print(f"Total Sentences : {len(lengths)}")
            print(f"Vocab Size      : {self.vocab_size}")
            print(f"Min Seq Length  : {min(lengths)}")
            print(f"Max Seq Length  : {max(lengths)}")
            print(f"Mean Seq Length : {sum(lengths) / len(lengths):.2f}")
            print("="*30 + "\n")

    def _update_reverse_vocab(self):
        self.idx2word = {int(i): w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text, add_special=True):
        clean_text = self._clean_text(text)
        units = self._tokenize(clean_text)
        indices = [self.word2idx.get(u, self.word2idx[self.unk_token]) for u in units]
        
        if add_special:
            indices = [self.word2idx[self.bos_token]] + indices + [self.word2idx[self.eos_token]]
            
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        """Converts indices back to string. Handles join logic correctly."""
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        
       
        units = [self.idx2word.get(int(i), self.unk_token) for i in indices]
        
        if self.level == "word":
            return " ".join(units)
        
        
        return "".join(units)

    def save(self, filepath):
       
        data = {
            "level": self.level,
            "word2idx": self.word2idx
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
       
        if isinstance(data, dict) and "level" in data:
            self.level = data["level"]
            self.word2idx = data["word2idx"]
        else:
            self.word2idx = data # Backward compatibility
            
        self._update_reverse_vocab()