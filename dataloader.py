import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class ASRSampledDataset(Dataset):
    def __init__(self, data_dir, num_indices=64):
        self.data_dir = Path(data_dir)
        self.file_list = list(self.data_dir.glob("*.pt"))
        self.num_indices = num_indices

    def __len__(self):
        return len(self.file_list)

    def _get_random_indices(self, features):
        seq_len = features.size(0)
        if seq_len > self.num_indices:
            indices = sorted(random.sample(range(seq_len), self.num_indices))
            return features[indices, :]
        return features

    def __getitem__(self, index):
        data_bundle = torch.load(self.file_list[index], weights_only=True)
        sampled_features = self._get_random_indices(data_bundle["features"])
        return sampled_features, data_bundle["text"]

def collate_fn(batch, pad_to_fixed_target=True, target_len=64):
    """
    Collate function with conditional padding logic.
    If pad_to_fixed_target is True, it ensures all samples are (target_len, 768).
    Otherwise, it pads to the max length found in the current batch.
    """
    features, transcripts = zip(*batch)
    
    if pad_to_fixed_target:
        padded_list = []
        for feat in features:
            pad_amt = target_len - feat.size(0)
            if pad_amt > 0:
                feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_amt), value=0.0)
            padded_list.append(feat)
        features_padded = torch.stack(padded_list)
    else:
        features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    return features_padded, list(transcripts)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "./processed_data"
    
   
    try:
        dataset = ASRSampledDataset(DATA_PATH, num_indices=64)
        print(f"Dataset initialized with {len(dataset)} samples.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run your preprocessing script first.")
        exit()

    
    # collate_loader = DataLoader(
    #     dataset, 
    #     batch_size=8, 
    #     shuffle=True, 
    #     collate_fn=lambda b: collate_fn(b, pad_to_fixed_target=True, target_len=64)
    # )

    loader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, pad_to_fixed_target=True, target_len=64)
    )

   
    print(f"Moving batch to: {device}\n" + "-"*30)
    
    for i, (features, texts) in enumerate(loader):
        
        features = features.to(device)
        
  
        print(f"Batch {i+1} Info:")
        print(f"  Feature Tensor Shape: {features.shape}") # Expect [8, 64, 768]
        print(f"  Transcription Count:  {len(texts)}")
        print(f"  First text in batch:  '{texts[0][:50]}...'")
        break