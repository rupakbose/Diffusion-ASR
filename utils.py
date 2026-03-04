import torch
from typing import List, Tuple, Any, Iterable

def preprocess_text_batch(raw_texts, tokenizer, device, target_len=256): # Increased for Char level
    """Tokenizes and pads text to a fixed target length."""
    encoded_list = [tokenizer.encode(t) for t in raw_texts]
    
    # Get the PAD ID once
    pad_id = tokenizer.word2idx[tokenizer.pad_token]
    
    text_tokens = torch.nn.utils.rnn.pad_sequence(
        encoded_list,
        batch_first=True,
        padding_value=pad_id
    ).to(device)

    current_len = text_tokens.size(1)
    if current_len < target_len:
        pad_amt = target_len - current_len
        text_tokens = torch.nn.functional.pad(
            text_tokens, (0, pad_amt), 
            value=pad_id
        )
    else:
        # If character level, this truncate is very likely to happen if target_len is 70
        text_tokens = text_tokens[:, :target_len]
        
    return text_tokens


def decode_sequences(
    logits: torch.Tensor, 
    tokenizer: Any, 
    special_tokens_to_filter: Iterable[str], 
    label_tokens: torch.Tensor
) -> Tuple[List[str], List[str]]:
    """Decodes model logits and ground truth tokens into clean text strings."""

    def _tokens_to_text(token_ids: Iterable[Iterable[int]]) -> List[str]:
        decoded_batch = []
        for sequence in token_ids:
            # Convert IDs to words/chars
            units = [tokenizer.idx2word.get(int(idx), tokenizer.unk_token) for idx in sequence]
            # Filter out special tokens ([PAD], [CLS], etc.)
            clean_units = [u for u in units if u not in special_tokens_to_filter]
            
            # Join logic: usually "" for char-level, " " for word-level
            # Keeping your " ".join() logic as requested
            decoded_batch.append(" ".join(clean_units))
        return decoded_batch

    # Extract predicted IDs (Argmax is enough, Softmax is redundant for selection)
    predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    
    # Process both predictions and labels
    decoded_labels = _tokens_to_text(label_tokens)
    decoded_preds = _tokens_to_text(predictions)

    return decoded_labels, decoded_preds