import os
import mne
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from scipy.io.wavfile import write
import textgrids

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["segment_path"], sr=16_000)
    # batch["sentence"] = batch["sentence"].upper()
    return speech_array

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

if __name__ == '__main__':

    dir_data = 'Dados/PEAC_Ativo - Análise - Aline e André'

    participant = 'Isabelle Vasconcelos'

    mne_eeg_file = "isabelle Vasconceles PEAC R ativo.vhdr"
    mne_anot_file = "isabelle Vasconceles PEAC R ativo.vmrk"
    audio_wav_file = "isabelle Vasconceles PEAC R ativo.wav"
    audio_seg_wav_file = "wav/isabelle Vasconceles PEAC R ativo_segm{}.wav"
    segm_csv_file = "isabelle Vasconceles PEAC R ativo audio wav segments.csv"
    audio_textgrid_file = "isabelle Vasconceles PEAC R ativo.TextGrid"

    mne_eeg_path = os.path.join(dir_data, participant, mne_eeg_file)
    mne_anot_path = os.path.join(dir_data, participant, mne_anot_file)
    audio_wav_path = os.path.join(dir_data, participant, audio_wav_file)
    audio_seg_wav_path = os.path.join(dir_data, participant, 'wav', audio_seg_wav_file)
    segm_csv_path = os.path.join(dir_data, participant, segm_csv_file)
    audio_textgrid_path = os.path.join(dir_data, participant, audio_textgrid_file)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    raw = mne.io.read_raw_brainvision(
        mne_eeg_path,
        preload=False,
        verbose=False
    )

    raw_audio = raw.pick(['audio'])

    del raw

    raw_audio.resample(16000)

    sampling_freq = raw_audio.info['sfreq']

    annotations = mne.read_annotations(
        mne_anot_path
    )

    annotations_df = annotations.to_data_frame()

    onset_arr = annotations_df['onset'].to_numpy()

    start_value = (onset_arr[1]-onset_arr[0]).astype('timedelta64[ms]').astype(np.int32)

    list_interval_ms = []
    for ind in range(1,onset_arr.size-1):
        list_interval_ms.append((onset_arr[ind+1]-onset_arr[ind]).astype('timedelta64[ms]').astype(np.int32))

    list_samples_pos = [(sum(list_interval_ms[:ind+1])+start_value - 100) for ind, val in enumerate(list_interval_ms)]

    list_samples_pos.insert(0,start_value-100)

    arr_samples_pos = np.array(list_samples_pos)

    audio = raw_audio.get_data(picks=['audio'])[0]

    write(audio_wav_path, int(sampling_freq), audio)

    dict_df = {
        "xmin": [],
        "xmax": [],
        "type_seg":[],
        "segment_path": []
    }

    for ind, value in enumerate(list_samples_pos):
        if ind == 0:
            dict_df["xmin"].append(0)
            dict_df["xmax"].append(value/1000)
            dict_df["type_seg"].append('New Segment/')
            dict_df["segment_path"].append(np.nan)
        else:
            dict_df["xmin"].append(latest_value)
            dict_df["xmax"].append(value/1000)
            dict_df["type_seg"].append('Stimulus/S 1')
            audio_seg = raw_audio.get_data(picks=['audio'], tmin=latest_value, tmax=value/1000)[0]
            audio_seg_path = audio_seg_wav_path.format(ind)
            write(audio_seg_path, int(sampling_freq), audio_seg)
            dict_df["segment_path"].append(audio_seg_path)

        latest_value = value/1000

    df_audio_segm = pd.DataFrame().from_dict(dict_df)

    df_audio_segm_pred = df_audio_segm.iloc[1:, :]

    df_audio_segm_pred["speech"] = df_audio_segm_pred.apply(speech_file_to_array_fn, axis=1)

    pad_size = max([len(value) for value in df_audio_segm_pred["speech"]])

    df_audio_segm_pred["speech"] = df_audio_segm_pred["speech"].apply(padarray, size=pad_size)

    inputs = processor(df_audio_segm_pred["speech"].to_list(), sampling_rate=16_000, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)

    # for i in range(8):
    predicted_sentences.insert(0,np.nan)

    df_audio_segm["text"] = predicted_sentences

    df_audio_segm.to_csv(segm_csv_path)

    list_interval = []
    for ind, row in df_audio_segm.iterrows():
        if ind == 0:
            list_interval.append(textgrids.Interval(xmin=row['xmin'],xmax=row['xmax'], text="New Segment/"))
        else:
            list_interval.append(textgrids.Interval(xmin=row['xmin'],xmax=row['xmax'], text=row['text']))

    text_grid_file = textgrids.TextGrid()

    text_grid_file["phonemes"] = textgrids.Tier(list_interval)

    text_grid_file.write(audio_textgrid_path)

