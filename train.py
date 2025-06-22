import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import os

import config
import utils
from model import Encoder, Decoder, Seq2Seq, MultiHeadAttention

def custom_loss_function(predictions, targets, mask):
    # predictions: [batch_size, seq_len, 3]
    # targets: [batch_size, seq_len, 3]
    # mask: [batch_size, seq_len]

    pred_xy = predictions[:, :, :2]
    target_xy = targets[:, :, :2]
    pred_shape = predictions[:, :, 2]
    target_shape = targets[:, :, 2]
    
    mask_flat = mask.view(-1)
    pred_xy_flat = pred_xy.view(-1, 2)[mask_flat]
    target_xy_flat = target_xy.view(-1, 2)[mask_flat]
    pred_shape_flat = pred_shape.view(-1)[mask_flat]
    target_shape_flat = target_shape.view(-1)[mask_flat]
    
    if pred_xy_flat.numel() == 0:
        return torch.tensor(0.0, device=config.DEVICE, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    xy_loss = nn.L1Loss()(pred_xy_flat, target_xy_flat)
    shape_loss = nn.L1Loss()(pred_shape_flat, target_shape_flat)
    
    y_penalty = torch.tensor(0.0, device=config.DEVICE)
    seq_lengths = mask.sum(dim=1).long()
    valid_sequences_mask = seq_lengths > 0
    
    if valid_sequences_mask.any():
        seq_lengths_valid = seq_lengths[valid_sequences_mask] - 1
        batch_indices = torch.arange(predictions.size(0), device=config.DEVICE)[valid_sequences_mask]
        
        last_predicted_y = predictions[batch_indices, seq_lengths_valid, 1]
        target_y = torch.zeros_like(last_predicted_y)
        y_penalty = nn.L1Loss()(last_predicted_y, target_y)

    penalty_weight = 5.0
    total_loss = (10.0 * xy_loss) + (1.0 * shape_loss) + (penalty_weight * y_penalty)
    
    return total_loss, xy_loss, shape_loss, y_penalty

def train_epoch(model, dataloader, optimizer, criterion, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    epoch_xy_loss = 0
    epoch_shape_loss = 0
    epoch_y_penalty = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        encoder_features = batch['encoder_features'].to(config.DEVICE)
        encoder_phonemes = batch['encoder_phonemes'].to(config.DEVICE)
        decoder_input = batch['decoder_input'].to(config.DEVICE)
        target_output = batch['target_output'].to(config.DEVICE)
        target_mask = batch['target_mask'].to(config.DEVICE)

        optimizer.zero_grad()
        
        predictions = model(encoder_features, encoder_phonemes, decoder_input, teacher_forcing_ratio)
        
        loss, xy_loss, shape_loss, y_penalty = criterion(predictions, target_output, target_mask)

        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or Inf loss detected. Skipping batch.")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_xy_loss += xy_loss.item()
        epoch_shape_loss += shape_loss.item()
        epoch_y_penalty += y_penalty.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'xy': f'{xy_loss.item():.4f}', 'shape': f'{shape_loss.item():.4f}', 'y_pen': f'{y_penalty.item():.4f}'})

    return epoch_loss / len(dataloader), epoch_xy_loss / len(dataloader), epoch_shape_loss / len(dataloader), epoch_y_penalty / len(dataloader)

def evaluate_epoch(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_xy_loss = 0
    epoch_shape_loss = 0
    epoch_y_penalty = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            encoder_features = batch['encoder_features'].to(config.DEVICE)
            encoder_phonemes = batch['encoder_phonemes'].to(config.DEVICE)
            decoder_input = batch['decoder_input'].to(config.DEVICE)
            target_output = batch['target_output'].to(config.DEVICE)
            target_mask = batch['target_mask'].to(config.DEVICE)
            
            predictions = model(encoder_features, encoder_phonemes, decoder_input, teacher_forcing_ratio=0)
            
            loss, xy_loss, shape_loss, y_penalty = criterion(predictions, target_output, target_mask)
            
            epoch_loss += loss.item()
            epoch_xy_loss += xy_loss.item()
            epoch_shape_loss += shape_loss.item()
            epoch_y_penalty += y_penalty.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'xy': f'{xy_loss.item():.4f}', 'shape': f'{shape_loss.item():.4f}', 'y_pen': f'{y_penalty.item():.4f}'})

    return epoch_loss / len(dataloader), epoch_xy_loss / len(dataloader), epoch_shape_loss / len(dataloader), epoch_y_penalty / len(dataloader)

def plot_predictions(model, dataset, epoch, num_examples=20):
    model.eval()
    
    fig, axs = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
    fig.suptitle(f'Validation Predictions at Epoch {epoch+1}', fontsize=16)

    with torch.no_grad():
        sample_indices = np.random.choice(len(dataset), num_examples, replace=False)
        
        for i, sample_idx in enumerate(sample_indices):
            sample = dataset[sample_idx]
            
            encoder_features = sample['encoder_features'].unsqueeze(0).to(config.DEVICE)
            encoder_phonemes = sample['encoder_phonemes'].unsqueeze(0).to(config.DEVICE)
            decoder_input = sample['decoder_input'].unsqueeze(0).to(config.DEVICE)
            target_output = sample['target_output']
            
            prediction = model(encoder_features, encoder_phonemes, decoder_input, teacher_forcing_ratio=0).squeeze(0).cpu().numpy()
            
            true_len = len(target_output)
            target_output = target_output.numpy()
            prediction = prediction[:true_len]
            
            axs[i, 0].plot(target_output[:, 0], label='Target X')
            axs[i, 0].plot(prediction[:, 0], label='Predicted X', linestyle='--')
            axs[i, 0].legend()
            if i == 0: axs[i, 0].set_title('X Coordinate')
            
            axs[i, 1].plot(target_output[:, 1], label='Target Y')
            axs[i, 1].plot(prediction[:, 1], label='Predicted Y', linestyle='--')
            axs[i, 1].legend()
            if i == 0: axs[i, 1].set_title('Y Coordinate')

            axs[i, 2].plot(target_output[:, 2], label='Target Shape', marker='o')
            axs[i, 2].plot(prediction[:, 2], label='Predicted Shape', linestyle='--', marker='x')
            axs[i, 2].legend()
            if i == 0: axs[i, 2].set_title('Shape')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"lstm_epoch_{epoch+1}_plot.png")
    plt.close(fig)
    print(f"Plot with {num_examples} examples saved to lstm_epoch_{epoch+1}_plot.png")


def main():
    print("Parsing USTX files...")
    all_notes = utils.parse_ustx_files(config.USTX_DIR)
    
    print("Building vocabulary...")
    phoneme_to_idx, idx_to_phoneme = utils.build_vocab(all_notes)
    utils.save_vocabulary(config.VOCAB_PATH, phoneme_to_idx, idx_to_phoneme)
    
    dataset = utils.PitchDataset(all_notes, phoneme_to_idx, config.SEQUENCE_LEN)
    
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=utils.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=utils.collate_fn)

    attention_layer = MultiHeadAttention(config.ENCODER_HIDDEN_DIM, config.DECODER_HIDDEN_DIM, config.ATTENTION_HEADS).to(config.DEVICE)
    encoder = Encoder(
        phoneme_vocab_size=len(phoneme_to_idx),
        phoneme_embedding_dim=config.PHONEME_EMBEDDING_DIM,
        note_feature_dim=config.NOTE_FEATURE_DIM,
        hidden_dim=config.ENCODER_HIDDEN_DIM,
        num_layers=config.ENCODER_NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    decoder = Decoder(
        output_dim=3, # x, y, shape
        hidden_dim=config.DECODER_HIDDEN_DIM,
        num_layers=config.DECODER_NUM_LAYERS,
        dropout=config.DROPOUT,
        attention=attention_layer
    ).to(config.DEVICE)

    model = Seq2Seq(encoder, decoder).to(config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')
    initial_tf_ratio = 1.0
    final_tf_ratio = 0.1
    tf_decay_steps = config.EPOCHS * 0.75

    for epoch in range(config.EPOCHS):
        if epoch < tf_decay_steps:
            teacher_forcing_ratio = initial_tf_ratio - (initial_tf_ratio - final_tf_ratio) * (epoch / tf_decay_steps)
        else:
            teacher_forcing_ratio = final_tf_ratio

        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} (Teacher Forcing: {teacher_forcing_ratio:.2f}) ---")
        
        train_loss, train_xy, train_shape, train_y_pen = train_epoch(model, train_loader, optimizer, custom_loss_function, teacher_forcing_ratio)
        print(f"  Train -> Loss: {train_loss:.4f} (XY: {train_xy:.4f}, Shape: {train_shape:.4f}, Y-Penalty: {train_y_pen:.4f})")
        
        val_loss, val_xy, val_shape, val_y_pen = evaluate_epoch(model, val_loader, custom_loss_function)
        print(f"  Valid -> Loss: {val_loss:.4f} (XY: {val_xy:.4f}, Shape: {val_shape:.4f}, Y-Penalty: {val_y_pen:.4f})")

        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  New best model saved with val_loss: {best_val_loss:.4f}")
            
        if epoch + 1 >= 30:
            checkpoint_path = os.path.join(checkpoint_dir, f"autopitch_lstm_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")

        if (epoch + 1) % config.PLOT_EVERY == 0:
            plot_predictions(model, val_dataset, epoch)

    print("\nTraining finished.")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main() 