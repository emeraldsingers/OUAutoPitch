# encoding: utf-8
import glob
import os
import re
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import bisect

import config

def normalize_phoneme(phoneme: str):
    try:
        if " " in phoneme:
            phoneme = phoneme.split(" ")[1]
        if phoneme not in ["息", "吸", '・', "R"]:
            phoneme = re.sub(r"[A-QS-Z0-9・ '\＃弱息↓捨カ囁強↑裏^深浅_\[\]]", "", phoneme, flags=re.UNICODE)
        return phoneme
    except TypeError:
        return None

def build_vocab(all_notes):
    phonemes = sorted(list(set(note['lyric'] for note in all_notes)))
    phoneme_to_idx = {p: i+2 for i, p in enumerate(phonemes)}
    phoneme_to_idx['<pad>'] = 0
    phoneme_to_idx['<unk>'] = 1
    idx_to_phoneme = {i: p for p, i in phoneme_to_idx.items()}
    return phoneme_to_idx, idx_to_phoneme

def save_vocabulary(vocab_path, phoneme_to_idx, idx_to_phoneme):
    with open(vocab_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'phoneme_to_idx': phoneme_to_idx,
            'idx_to_phoneme': idx_to_phoneme
        }, f, allow_unicode=True)
    print(f"Vocabulary saved to {vocab_path}")

def load_vocabulary(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = yaml.safe_load(f)
    return vocab_data['phoneme_to_idx'], vocab_data['idx_to_phoneme']

def load_ustx(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Could not load {file_path}: {e}")
        return None

def get_tempo_at_position(tempos, position):
    if not tempos:
        return 120.0
    
    tempo_positions = [t['position'] for t in tempos]
    idx = bisect.bisect_right(tempo_positions, position) - 1
    if idx < 0:
        idx = 0
    return tempos[idx]['bpm']

def position_to_seconds(position, duration, tempos):
    if not tempos:
        return duration / 1920.0 * (60.0 / 120.0)
    
    end_position = position + duration
    current_position = position
    total_seconds = 0.0
    
    while current_position < end_position:
        tempo_positions = [t['position'] for t in tempos]
        idx = bisect.bisect_right(tempo_positions, current_position) - 1
        if idx < 0:
            idx = 0
        
        current_tempo = tempos[idx]['bpm']
        
        if idx + 1 < len(tempos) and tempos[idx + 1]['position'] < end_position:
            next_position = tempos[idx + 1]['position']
            segment_duration = next_position - current_position
            total_seconds += segment_duration / 1920.0 * (60.0 / current_tempo)
            current_position = next_position
        else:
            segment_duration = end_position - current_position
            total_seconds += segment_duration / 1920.0 * (60.0 / current_tempo)
            break
    
    return total_seconds

def parse_ustx_files(ustx_dir):
    ustx_files = glob.glob(os.path.join(ustx_dir, "*.ustx"))
    print(f"Found {len(ustx_files)} USTX files.")

    all_notes = []
    cpu_count = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=cpu_count//2 or 1) as executor:
        results = list(tqdm(executor.map(load_ustx, ustx_files), total=len(ustx_files), desc="Loading USTX files"))

    for data in tqdm(results, desc="Extracting notes"):
        if data is None or "voice_parts" not in data:
            continue
            
        tempos = data.get("tempos", [{"position": 0, "bpm": 120}])
            
        for part in data["voice_parts"]:
            track_notes = []
            if "notes" not in part:
                continue
            for note in part["notes"]:
                norm_lyric = normalize_phoneme(note["lyric"])
                if not norm_lyric:
                    continue
                
                seconds = position_to_seconds(note["position"], note["duration"], tempos)
                
                note_info = {
                    "position": note["position"],
                    "duration": note["duration"],
                    "seconds": seconds,
                    "tone": note["tone"],
                    "lyric": norm_lyric,
                    "pitch": note["pitch"]["data"],
                    "tempos": tempos
                }
                track_notes.append(note_info)
            track_notes.sort(key=lambda n: n['position'])
            all_notes.extend(track_notes)
            
    print(f"Total notes extracted: {len(all_notes)}")
    return all_notes


def _extract_note_features(notes_in_sequence, target_note_index):
    features = [0.0] * config.NOTE_FEATURE_DIM
    current_note = notes_in_sequence[target_note_index]
    
    features[0] = current_note["seconds"] / 10.0      
    features[5] = current_note["tone"] / 127.0

    if target_note_index > 0:
        prev_note = notes_in_sequence[target_note_index - 1]
        features[1] = (current_note["tone"] - prev_note["tone"]) / 12.0
        
        prev_end_pos = prev_note['position'] + prev_note['duration']
        gap_ticks = max(0, current_note['position'] - prev_end_pos)
        
        if gap_ticks > 0:
            tempos = current_note.get('tempos', [{"position": 0, "bpm": 120}])
            gap_seconds = position_to_seconds(prev_end_pos, gap_ticks, tempos)
            features[2] = gap_seconds / 1.0
        else:
            features[2] = 0.0
    
    if target_note_index < len(notes_in_sequence) - 1:
        next_note = notes_in_sequence[target_note_index + 1]
        features[3] = (next_note["tone"] - current_note["tone"]) / 12.0
        
        current_end_pos = current_note['position'] + current_note['duration']
        gap_ticks = max(0, next_note['position'] - current_end_pos)
        
        if gap_ticks > 0:
            tempos = current_note.get('tempos', [{"position": 0, "bpm": 120}])
            gap_seconds = position_to_seconds(current_end_pos, gap_ticks, tempos)
            features[4] = gap_seconds / 1.0
        else:
            features[4] = 0.0
        
    return features

def _extract_pitch_params(note: dict):
    pitch_points = note['pitch']
    
    params = [[0.0, 0.0, -2.0]]  # <bos> token
    
    for point in pitch_points:
        shape_map = {'i': -1.0, 'o': 1.0, 'io': 0.0}
        shape = shape_map.get(point.get('shape', 'io'), 0.0)
        x = point['x'] / 100.0
        y = point['y'] / 100.0
        params.append([x, y, shape])
    
    params.append([0.0, 0.0, 2.0])  # <eos> token
    return params

class PitchDataset(Dataset):
    def __init__(self, all_notes, phoneme_to_idx, sequence_len):
        self.all_notes = all_notes
        self.phoneme_to_idx = phoneme_to_idx
        self.sequence_len = sequence_len
        self.half_seq = sequence_len // 2

    def __len__(self):
        return len(self.all_notes)

    def __getitem__(self, index):
        start = max(0, index - self.half_seq)
        end = min(len(self.all_notes), index + self.half_seq + 1)
        
        note_window = self.all_notes[start:end]
        
        target_note_in_window_idx = index - start
        
        encoder_note_features = []
        encoder_phoneme_ids = []

        for i, note in enumerate(note_window):
            features = _extract_note_features(note_window, i)
            encoder_note_features.append(features)
            
            phoneme_id = self.phoneme_to_idx.get(note['lyric'], self.phoneme_to_idx['<unk>'])
            encoder_phoneme_ids.append(phoneme_id)
        
        target_note = note_window[target_note_in_window_idx]
        pitch_params = _extract_pitch_params(target_note)
        
        decoder_input = [p for p in pitch_params[:-1]]
        target_output = [p for p in pitch_params[1:]]
        
        return {
            'encoder_features': torch.FloatTensor(encoder_note_features),
            'encoder_phonemes': torch.LongTensor(encoder_phoneme_ids),
            'decoder_input': torch.FloatTensor(decoder_input),
            'target_output': torch.FloatTensor(target_output)
        }

def collate_fn(batch):
    encoder_features = [item['encoder_features'] for item in batch]
    encoder_phonemes = [item['encoder_phonemes'] for item in batch]
    
    padded_encoder_features = torch.nn.utils.rnn.pad_sequence(encoder_features, batch_first=True, padding_value=0.0)
    padded_encoder_phonemes = torch.nn.utils.rnn.pad_sequence(encoder_phonemes, batch_first=True, padding_value=0)

    decoder_inputs = [item['decoder_input'] for item in batch]
    target_outputs = [item['target_output'] for item in batch]
    
    padded_decoder_inputs = torch.nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=0.0)
    padded_target_outputs = torch.nn.utils.rnn.pad_sequence(target_outputs, batch_first=True, padding_value=0.0)

    target_mask = (padded_target_outputs.sum(dim=-1) != 0)

    return {
        'encoder_features': padded_encoder_features,
        'encoder_phonemes': padded_encoder_phonemes,
        'decoder_input': padded_decoder_inputs,
        'target_output': padded_target_outputs,
        'target_mask': target_mask
    } 