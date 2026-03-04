import torch
import yaml
import torch.nn.functional as F
from pathlib import Path
from rich.console import Console
from rich.live import Live
from rich.table import Table
import torchaudio
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import random
from jiwer import cer, wer

# Local Imports
from tokenizer import WordTokenizer
from models.model import Transformer

def get_ground_truth(audio_path):
    audio_path = Path(audio_path)
    file_id = audio_path.stem  
    trans_files = list(audio_path.parent.glob("*.trans.txt"))
    
    if not trans_files:
        return "No .trans.txt found in folder."
        
    for trans_file in trans_files:
        with open(trans_file, 'r') as f:
            for line in f:
                if line.startswith(file_id):
                    
                    return line.strip().split(" ", 1)[1]
                    
    return f"ID {file_id} not found in transcript files."

def sample_audio_features(features, target_len=64):
    """
    Applies the same sorted random index sampling used in training.
    Shape input: [1, T, 768] -> Output: [1, target_len, 768]
    """
    seq_len = features.size(1)
    
    if seq_len > target_len:
        indices = sorted(random.sample(range(seq_len), target_len))
        sampled_features = features[:, indices, :]
    elif seq_len < target_len:
        pad_amt = target_len - seq_len
        sampled_features = torch.nn.functional.pad(features, (0, 0, 0, pad_amt), value=0.0)
    else:
        sampled_features = features
        
    return sampled_features

def load_model(checkpoint_path, config, vocab_size, device):
    model = Transformer(
        tgt_vocab_size=vocab_size,
        d_model=config["projection_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_enc"],
        d_ff=config["feedforward_dim"],
        max_seq_length=config["transcription_len"] + config["window_size"],
        dropout=config["dropout"]
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

@torch.no_grad()
def run_inference(audio_path, checkpoint_path, config_path="config.yaml", steps = 100):

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Config and Tokenizer
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = WordTokenizer()
    tokenizer.load(Path(checkpoint_path).parent / "tokenizer.json")

    mask_id = tokenizer.word2idx['<MASK>']
    vocab_size = tokenizer.vocab_size

    model = load_model(checkpoint_path, config, vocab_size, device)
    target_len = config["transcription_len"]
    tokens = torch.full((1, target_len), mask_id, dtype=torch.long).to(device)
    
    
    console.print(f"[bold green]Starting Denoising Inference...[/bold green]")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    modelAudio = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    waveform, sr = torchaudio.load(audio_path)              
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        raw_features = modelAudio(**inputs).last_hidden_state
    audio_features = sample_audio_features(raw_features, target_len=config["window_size"])
    ground_truth = get_ground_truth(audio_path)

    # Initial state: everything is masked
    # mask = torch.ones((1, target_len), dtype=torch.bool, device=device)
    tokens = torch.full((1, target_len), mask_id, dtype=torch.long, device=device)

    times = torch.linspace(1, 0, steps + 1, device=device)
    s1=0
    with Live(console=console, refresh_per_second=4) as live:
        for i, (t, s) in enumerate(zip(times[:-1], times[1:])):
            output = model(audio_features, tokens)
            logits = output['text_token'] # [1, Seq, Vocab]

            probs_all = torch.nn.functional.softmax(logits, dim=-1)
            
            sampled_tokens = torch.multinomial(probs_all[0], num_samples=1).squeeze(-1)
            tokens = sampled_tokens.unsqueeze(0) # [1, Seq]
            token_units = [tokenizer.idx2word.get(int(i), tokenizer.unk_token) for i in tokens.squeeze().tolist()]
            remask_probs = torch.rand_like(tokens , dtype = torch.float, device = device)
            remask_probs = (remask_probs < s/t)
            tokens[remask_probs] =  mask_id
            

            special_to_filter = {
                tokenizer.pad_token, 
                tokenizer.bos_token, 
                tokenizer.eos_token, 
                tokenizer.mask_token, 
                tokenizer.unk_token
            }

            clean_units = [u for u in token_units if u not in special_to_filter]
            
            if tokenizer.level == "word":
                decoded_text = " ".join(clean_units)
            else:
                decoded_text = "".join(clean_units)

            
            table = Table(title=f"Step {s1}/{steps}", box=None)
            table.add_column(f"Iterative Refinement ({tokenizer.level.upper()})", style="bold cyan")
            
            
            display_text = decoded_text if decoded_text.strip() else "[dim]...denoising...[/dim]"
            table.add_row(display_text)
            live.update(table)
            
            s1 += 1

    console.print("\n" + "━" * 60)
    ground_truth_clean = ground_truth.strip().upper()
    final_result = decoded_text.strip().upper()
    Char_error = cer(ground_truth_clean, final_result)
    Word_error = wer(ground_truth_clean, final_result)

    console.print("\n" + "━" * 60)
    console.print(f"[bold cyan]GROUND TRUTH :[/bold cyan] {ground_truth_clean}")
    console.print(f"[bold green]MODEL OUTPUT :[/bold green] {final_result}")

    
    console.print(f"[bold red] CER: {Char_error}, WER: {Word_error})")
    console.print("━" * 60 + "\n")

if __name__ == "__main__":
    CHECKPOINT = "./checkpoints/20260304_1154_affectionate_bose/model_epoch_1100.pt"
    SAMPLE_AUDIO = "./Dataset/dev-clean-2/5895/34615/5895-34615-0000.flac"
    STEPS = 10
    run_inference( audio_path= SAMPLE_AUDIO, checkpoint_path= CHECKPOINT, steps= STEPS)