import torch
import yaml
import numpy as np
import random
from tqdm import tqdm

import config
import utils
from model import Encoder, Decoder, Seq2Seq, MultiHeadAttention

def predict_pitch_curve(model, note_context_window, phoneme_to_idx):
    """
    Predicts the pitch curve for a single target note given its context.
    """
    model.eval()

    encoder_note_features = []
    encoder_phoneme_ids = []
    
    target_note_in_window_idx = len(note_context_window) // 2

    for i, note in enumerate(note_context_window):
        features = utils._extract_note_features(note_context_window, i)
        encoder_note_features.append(features)
        
        phoneme_id = phoneme_to_idx.get(note['lyric'], phoneme_to_idx['<unk>'])
        encoder_phoneme_ids.append(phoneme_id)
        
    encoder_features_tensor = torch.FloatTensor(encoder_note_features).unsqueeze(0).to(config.DEVICE)
    encoder_phonemes_tensor = torch.LongTensor(encoder_phoneme_ids).unsqueeze(0).to(config.DEVICE)
    
    target_note = note_context_window[target_note_in_window_idx]
    target_duration_tensor = torch.FloatTensor([target_note['duration'] / 1920.0]).to(config.DEVICE)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(encoder_features_tensor, encoder_phonemes_tensor)
        
        current_input_point = torch.FloatTensor([[0.0, 0.0, -2.0]]).unsqueeze(1).to(config.DEVICE)
        
        duration_tensor = target_duration_tensor.unsqueeze(1).unsqueeze(1).expand(-1, 1, 1)

        predicted_points = []
        for _ in range(config.MAX_PITCH_POINTS):
            current_input_with_duration = torch.cat([current_input_point, duration_tensor], dim=2)
            output, hidden, cell = model.decoder(current_input_with_duration, hidden, cell, encoder_outputs)
            current_input_point = output
            predicted_point = output.squeeze(0).squeeze(0).cpu().numpy()
            
            if predicted_point[2] > 1.5:
                break
                
            predicted_points.append(predicted_point)
            print(len(predicted_points))
    return predicted_points

def convert_predictions_to_ustx_format(predicted_points, note_duration):
    """
    Converts the raw model predictions back to the USTX pitch data format,
    ensuring the curve ends at y=0 at the note's duration.
    """
    new_pitch_data = []
    for point in predicted_points:
        x_val, y_val, shape_val = point[0], point[1], point[2]
        
        shape_str = "io"
        if shape_val < -0.5: shape_str = "i"
        elif shape_val > 0.5: shape_str = "o"
            
        new_pitch_data.append({
            "x": float(x_val * 100.0),
            "y": float(y_val * 100.0),
            "shape": shape_str
        })
        
    valid_points = [p for p in new_pitch_data if p['x'] < note_duration]

    if not valid_points:
        return [{"x": float(note_duration), "y": 0.0, "shape": "io"}]
    base_points = [p for p in valid_points if p['x']]
    
    if not base_points and valid_points:
        base_points = [valid_points[-1]]

    final_points_map = {p['x']: p for p in base_points}

    new_pitch_data = sorted(list(final_points_map.values()), key=lambda p: p["x"])
        
    return new_pitch_data

def main(ustx_path, output_path):
    print("Loading vocabulary...")
    phoneme_to_idx, _ = utils.load_vocabulary(config.VOCAB_PATH)

    print("Initializing model...")
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
        output_dim=3,
        hidden_dim=config.DECODER_HIDDEN_DIM,
        num_layers=config.DECODER_NUM_LAYERS,
        dropout=config.DROPOUT,
        attention=attention_layer
    ).to(config.DEVICE)

    model = Seq2Seq(encoder, decoder).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    print(f"Model loaded from {config.MODEL_SAVE_PATH}")
    
    print(f"Loading and parsing {ustx_path}...")
    ustx_data = utils.load_ustx(ustx_path)
    if not ustx_data:
        return
        
    all_notes = []
    for part in ustx_data.get("voice_parts", []):
        track_notes = []
        for note in part.get("notes", []):
            norm_lyric = utils.normalize_phoneme(note["lyric"])
            if not norm_lyric:
                continue
            track_notes.append({
                "position": note["position"],
                "duration": note["duration"],
                "tone": note["tone"],
                "lyric": norm_lyric,
                "original_note": note
            })
        track_notes.sort(key=lambda n: n['position'])
        all_notes.extend(track_notes)

    if not all_notes:
        print("No processable notes found in the USTX file.")
        return

    print(f"Predicting pitches for {len(all_notes)} notes...")
    half_seq = config.SEQUENCE_LEN // 2
    
    for i in tqdm(range(len(all_notes)), desc="Predicting Pitch Curves"):
        start = max(0, i - half_seq)
        end = min(len(all_notes), i + half_seq + 1)
        note_window = all_notes[start:end]
        
        predicted_points = predict_pitch_curve(model, note_window, phoneme_to_idx)
        
        target_note_duration = all_notes[i]["duration"]
        new_pitch_data = convert_predictions_to_ustx_format(predicted_points, target_note_duration)
        
        if new_pitch_data:
            all_notes[i]["original_note"]["pitch"]["data"] = new_pitch_data
        else:
            pass 
            
    print(f"Saving tuned output to {output_path}...")
    ustx_data['comment'] = "Tuned by AutoPitch made by asoqwer (Emerald Project)"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('name: Merged Project\n')
        f.write('output_dir: Vocal\n')
        f.write('cache_dir: UCache\n')
        f.write('ustx_version: "0.6"\n')
        f.write('resolution: 480\n')
        f.write('bpm: 120\n')
        f.write('beat_per_bar: 4\n')
        yaml.dump(ustx_data, f, allow_unicode=True)
    print("Inference complete.")

if __name__ == '__main__':
    import argparse
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description="Use a trained AutoPitch LSTM model to tune a USTX file.")
    parser.add_argument("input_ustx", type=str, help="Path to the input USTX file.")
    parser.add_argument("output_ustx", type=str, help="Path to save the tuned output USTX file.")
    args = parser.parse_args()
    
    main(args.input_ustx, args.output_ustx)