# AutoPitch LSTM

A deep learning-based pitch curve generator for UTAU `.ustx` files, using a bidirectional LSTM encoder and multi-head attention decoder. This project is designed for creative, robust, and musically intelligent pitch curve generation.

## Features
- **Bidirectional LSTM Encoder**: Understands both past and future note context.
- **Multi-Head Attention Decoder**: Allows the model to focus on the most relevant notes for each pitch point.
- **Custom Loss**: Teaches the model to end pitch curves at the baseline (`y=0`).
- **Smart Inference**: Ensures all notes end musically and naturally.
- **Checkpointing**: Saves model snapshots every epoch (from epoch 30) for easy rollback and experimentation.
- **Reproducibility**: All random seeds are fixed for consistent results.

## Installation
1. **Clone the repository** and `cd` into the project folder.
2. **Install dependencies**:
   ```bash
   pip install torch numpy pyyaml matplotlib tqdm
   ```
3. **Prepare your data**:
   - Place your `.ustx` files in the `ustx/` directory.

## Training
Train the model on your `.ustx` dataset:
```bash
python train.py
```
- Training progress, validation plots, and checkpoints will be saved automatically.
- The best model is saved as `autopitch_lstm_best.pth`.
- Checkpoints for each epoch (from 30 onward) are saved in the `checkpoints/` folder.

## Inference
Use your trained model to generate pitch curves for a new `.ustx` file:
```bash
python infer.py path/to/input.ustx path/to/output.ustx
```
- The output `.ustx` will have tuned pitch curves.

## Configuration
All model and training parameters are in `config.py`:
- `SEQUENCE_LEN`: Context window size (number of notes before/after).
- `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, etc.
- `ATTENTION_HEADS`: Number of attention heads in the decoder.

## Tips for Best Results
- **More Data = Better Model**: The more `.ustx` files you provide, the better the model will learn.
- **Experiment with Checkpoints**: Try using models from different epochs for different creative results.
- **Monitor the Y-Penalty**: The custom loss will show a `Y-Penalty` term—lower is better for clean note endings.
- **Reproducibility**: Results are deterministic due to fixed seeds, but you can change them for more variety.

## Advanced
- The model uses a custom loss to encourage pitch curves to end at `y=0`.
- The inference script uses a smart "landing zone" to ensure musical endings.
- You can further tune the penalty weight in the loss function for stricter or looser endings.

## License
MIT License. See `LICENSE` file for details.

---

**Created with ❤️ for creative UTAU pitch automation.** 
