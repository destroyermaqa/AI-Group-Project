import torch
import torch.nn as nn
import torch.optim as optim
import time
import math

# Import components from your pipeline and vanilla model files
from pipeline import get_data_pipeline
from model_withoutAttention import Encoder, VanillaDecoder, Seq2Seq

def train_one_epoch(model, loader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    
    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (50% teacher forcing during training)
        output = model(src, trg, teacher_forcing_ratio=0.5)
        
        output_dim = output.shape[-1]
        
        # Reshape outputs and targets to fit CrossEntropyLoss (ignoring <SOS> at index 0)
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Clip gradients to prevent exploding gradients in the recurrent layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            
            # Turn off teacher forcing completely for evaluation
            output = model(src, trg, teacher_forcing_ratio=0.0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(loader)

def main():
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Active Device: {device}")
    
    # 2. Get data processing pipeline components
    BATCH_SIZE = 32
    print("Loading data tokenization and batching pipeline...")
    loader, en_vocab, de_vocab = get_data_pipeline(batch_size=BATCH_SIZE)
    
    # Dimensions based on your custom pipeline vocabulary lengths
    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(de_vocab)
    
    # Hyperparameters
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 1  
    DROPOUT = 0.5
    CLIP = 1.0
    EPOCHS = 5
    
    # 3. Model instantiation
    print("Initializing Vanilla (No-Attention) Seq2Seq Architecture...")
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = VanillaDecoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # 4. Optimization configurations
    PAD_IDX = de_vocab.stoi["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Training execution cycle
    print("\n--- Starting Vanilla Model Training ---")
    best_valid_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, loader, optimizer, criterion, CLIP, device)
        valid_loss = validate_one_epoch(model, loader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)
        
        # Save model checkpoint if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'vanilla_model_best.pt')
            
        print(f"Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\tVal. Loss:  {valid_loss:.3f} | Val. PPL:  {math.exp(valid_loss):7.3f}")

if __name__ == "__main__":
    main()
