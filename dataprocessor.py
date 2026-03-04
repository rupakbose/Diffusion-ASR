import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

def extract_features_safe(source_dir, output_dir):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    for transcript_path in tqdm(list(source_path.rglob("*.trans.txt"))):
        with open(transcript_path, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2: continue
                
                audio_id, text = parts
                audio_path = transcript_path.parent / f"{audio_id}.flac"

                if audio_path.exists():
                    waveform, sr = torchaudio.load(audio_path)
                    
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)

                    inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        features = model(**inputs).last_hidden_state.squeeze(0).cpu()

                    torch.save({"features": features, "text": text}, output_path / f"{audio_id}.pt")

if __name__ == "__main__":
    extract_features_safe("./Dataset/dev-clean-2", "/processed_data")