import os
import torch
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class BirdDataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=32000):
        self.filepaths = []
        self.labels = []
        self.target_sample_rate = target_sample_rate
        
        root_path = Path(root_dir)
        for file_path in root_path.rglob('*.ogg'):
            self.filepaths.append(str(file_path))
            self.labels.append(file_path.parent.name)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        
        np_waveform, _ = librosa.load(path, sr=self.target_sample_rate, mono=True)
        waveform = torch.from_numpy(np_waveform)
        
        return waveform, label, path

MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8"
model = hub.load(MODEL_URL)

audio_repertories = ["train_audio"]

for repertory_name in audio_repertories:
    audio_repertory = "ressource/raw_data/" + repertory_name
    dataset = BirdDataset(root_dir=audio_repertory)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    embeddings = []
    labels = []
    row_ids = []


    for batch_waveforms, batch_labels, batch_paths in tqdm(dataloader, desc="Extraction", unit="fichier"):
        try:
            audio_plat = batch_waveforms[0].numpy()
            tf_audio = tf.convert_to_tensor(audio_plat, dtype=tf.float32)
            
            frames = tf.signal.frame(tf_audio, frame_length=160000, frame_step=160000, pad_end=True)
            filename = Path(batch_paths[0]).stem
            
            for i in range(frames.shape[0]):
                frame_input = tf.expand_dims(frames[i], 0) 
                outputs = model.infer_tf(frame_input)
                
                if 'embeddings' in outputs:
                    emb = outputs['embeddings'].numpy()
                elif 'embedding' in outputs:
                    emb = outputs['embedding'].numpy()
                embeddings.append(np.squeeze(emb))
                
                if(repertory_name == "train_audio"):
                    labels.append(batch_labels[0])

                timestamp = (i + 1) * 5
                row_ids.append(f"{filename}_{timestamp}")
                
                

        except Exception as e:
            print(f"\nErreur sur {batch_paths[0]} : {e}")
            continue

    save_repertory = Path('ressource/processed_data4')
    save_repertory.mkdir(parents=True, exist_ok=True)

    np.save(save_repertory / f"X_embeddings_perch_{repertory_name}.npy", np.vstack(embeddings))
    
    if repertory_name == "train_audio":
        np.save(save_repertory / f"y_labels_{repertory_name}.npy", np.array(labels))
        
    np.save(save_repertory / f"row_ids_{repertory_name}.npy", np.array(row_ids))

    print(f"Data saved for {repertory_name} in {save_repertory} !")