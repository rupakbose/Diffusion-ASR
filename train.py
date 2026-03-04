import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from names_generator import generate_name
from tqdm import tqdm
from datetime import datetime
from jiwer import wer, cer


# Local Imports
from tokenizer import WordTokenizer
from dataloader import ASRSampledDataset, collate_fn
from models.model import Transformer
from utils import preprocess_text_batch, decode_sequences



def train():

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f"{timestamp}_{generate_name()}"
    print(f"Starting Experiment: {experiment_name}")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["log_dir"] = Path("runs") / experiment_name
    config["checkpoint_dir"] = Path("checkpoints") / experiment_name
    config["log_dir"].mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    with open(config["checkpoint_dir"] / "config_reference.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=str(config["log_dir"]))

    tokenizer = WordTokenizer(level=config["tokenization_level"])
    all_transcripts = list(Path("./Dataset/dev-clean-2").rglob("*.trans.txt"))
    tokenizer.fit(all_transcripts)
    tokenizer.save(config["checkpoint_dir"] / "tokenizer.json")
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.word2idx[tokenizer.pad_token]
    mask_token_id = tokenizer.word2idx['<MASK>']

    special_to_filter = {
                tokenizer.pad_token, 
                tokenizer.bos_token, 
                tokenizer.eos_token, 
                tokenizer.mask_token, 
                tokenizer.unk_token
            }
    
    dataset = ASRSampledDataset(config["data_dir"], num_indices=config["window_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers = config["num_workers"],
        prefetch_factor = config["prefetch_factor"],
        collate_fn=lambda b: collate_fn(b, pad_to_fixed_target=True, target_len=config["window_size"])
    )

    for batch_idx, (audio_features, raw_texts) in enumerate(loader):
        audio_features = audio_features.to(device)
        text_tokens = preprocess_text_batch(raw_texts, tokenizer, device, target_len=config["transcription_len"])
        print("\n" + "="*40)
        print(f"DATALOADAER for {experiment_name.upper()}")
        print("="*40)
        print(f"Audio Features Shape : {audio_features.shape}")
        print(f"Text Tokens Shape    : {text_tokens.shape}")
        print(f"First Text Sample    : {raw_texts[0][:50]}...")
        print(f"BOS/EOS/PAD IDs     : {tokenizer.word2idx[tokenizer.bos_token]}, "
              f"{tokenizer.word2idx[tokenizer.eos_token]}, {pad_idx}")
        print('len loader', len(loader))
        print("="*40)
        break

    
    model = Transformer(
                        tgt_vocab_size=vocab_size,
                        d_model=config["projection_dim"],
                        num_heads=config["num_heads"],
                        num_layers=config["num_enc"],
                        d_ff=config["feedforward_dim"],
                        max_seq_length=config["transcription_len"] + config["window_size"],
                        dropout=config["dropout"]
                    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    print(f"Vocab Size: {vocab_size} | Training on: {device}")
    global_step = 0

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        reference = []
        hypothesis = []


        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

        for batch_idx, (audio_features, raw_texts) in enumerate(pbar):
            audio_features = audio_features.to(device)
            batch_size = audio_features.shape[0]

            input_token_ids = preprocess_text_batch(raw_texts, tokenizer, device, target_len=config["transcription_len"])
            label_tokens = input_token_ids.clone().detach().cpu().numpy()

            t = torch.rand(batch_size, 1).to(device)
            t = t.expand(batch_size,input_token_ids.shape[-1]).clamp_min(1e-5)
            mask = torch.bernoulli(t).bool().to(device)
            masked_input_ids = input_token_ids.masked_fill(mask, mask_token_id)
            labels = input_token_ids.masked_fill(~mask, -100) #-100 is ignored by crossentropy as we want to compute loss 
           
           
            output = model(audio_features, masked_input_ids)
            logits = output['text_token']

            num_classes = logits.shape[-1]
            flat_logits = logits.reshape(-1, num_classes)
            flat_labels = labels.flatten()
            
            
            loss = loss_fn(flat_logits, flat_labels)
            loss = loss.reshape(batch_size, -1) / t 
            transcription_length = mask.sum(dim=1, keepdim=True)
            loss = loss / (transcription_length + 1)
            loss = loss.sum(dim=1).mean()

           
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})


            decoded_text_label, decoded_text_pred = decode_sequences(logits=logits,
                                                                     tokenizer=tokenizer,
                                                                     special_tokens_to_filter=special_to_filter,
                                                                     label_tokens=label_tokens)

            reference.extend(decoded_text_label)
            hypothesis.extend(decoded_text_pred)
            
            
            global_step += 1

        avg_epoch_loss = epoch_loss / len(loader)
        Char_error = cer(reference, hypothesis)
        Word_error = wer(reference, hypothesis)

        writer.add_scalar("Loss/train", avg_epoch_loss, epoch+1)
        writer.add_scalar("metric/CER", Char_error, epoch+1)
        writer.add_scalar("metric/WER", Word_error, epoch+1)
        print(f"Epoch {epoch+1}/{config['epochs']} - Avg Loss: {avg_epoch_loss:.4f}")
        
        if (epoch + 1) % config["save_every_epoch"] == 0:
            ckpt_name = f"model_epoch_{epoch+1}.pt"
            checkpoint_path = config["checkpoint_dir"] / ckpt_name
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'vocab_size': vocab_size
            }, checkpoint_path)

    writer.close()
    print(f"Training finished for experiment: {experiment_name}")


if __name__ == "__main__":
    train()