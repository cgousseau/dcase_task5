import argparse
import gzip
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import librosa

def extract_mel(annotation_path, dataset_dir, output_dir, progress=True):
    """
    Extract embeddings for files annotated in the SONYC annotation file and save them to disk.
    """

    print("* Loading annotations.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')

    out_dir = os.path.join(output_dir, 'mel')
    os.makedirs(out_dir, exist_ok=True)

    df = annotation_data[['split', 'audio_filename']].drop_duplicates()
    row_iter = df.iterrows()

    print("* Extracting embeddings.")
    for _, row in row_iter:
        filename = row['audio_filename']
        print('processing '+str(filename)+'...')
        split_str = row['split']
        audio_path = os.path.join(dataset_dir, split_str, filename)
        emb_path = os.path.join(out_dir, os.path.splitext(filename)[0])
        np.save(emb_path,compute_mel(audio_path,bands=64,frames=64))

def get_sound_data(path, sr=22050):
    data, fsr = sf.read(path)
    data_resample = librosa.resample(data.T, fsr, sr)
    if len(data_resample.shape) > 1:
        data_resample = np.average(data_resample, axis=0)
    return data_resample, sr

def compute_mel(file_name, bands=64, frames=64):
    
    window_size = 512 * (frames - 1)  
    class_labels = []    

    sound_data, sr = get_sound_data(file_name, sr=22050)

    signal = sound_data
    # get the log-scaled mel-spectrogram
    melspec_full = librosa.feature.melspectrogram(signal, n_mels = bands)
    logspec_full = librosa.amplitude_to_db(melspec_full)
    logspec_full = logspec_full.T.flatten()[:, np.newaxis].T
    # get the log-scaled, averaged values for the harmonic & percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(signal)
    melspec_harmonic = librosa.feature.melspectrogram(y_harmonic, n_mels = bands)
    melspec_percussive = librosa.feature.melspectrogram(y_percussive, n_mels = bands)
    logspec_harmonic = librosa.amplitude_to_db(melspec_harmonic)
    logspec_percussive = librosa.amplitude_to_db(melspec_percussive)
    logspec_harmonic = logspec_harmonic.T.flatten()[:, np.newaxis].T
    logspec_percussive = logspec_percussive.T.flatten()[:, np.newaxis].T
    logspec_hp = np.average([logspec_harmonic, logspec_percussive], axis=0)
                
    # create the first two feature maps    
    log_specgrams_full = np.array(logspec_full).reshape(np.array(logspec_full).shape[1]//bands ,bands, 1)
    log_specgrams_hp = np.array(logspec_hp).reshape(np.array(logspec_full).shape[1]//bands ,bands, 1)
    features = np.concatenate((log_specgrams_full, 
                               log_specgrams_hp, 
                               np.zeros(np.shape(log_specgrams_full))), 
                              axis=2)
    #create the third feature map which is the delta (derivative) of the log-scaled mel-spectrogram
    features[:,:,2] = librosa.feature.delta(features[:,:,0])
    
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("dataset_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--progress", action="store_const", const=True, default=False)

    args = parser.parse_args()

    extract_mel(annotation_path=args.annotation_path,
                              dataset_dir=args.dataset_dir,
                              output_dir=args.output_dir)
