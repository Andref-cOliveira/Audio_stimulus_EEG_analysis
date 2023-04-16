import os
import mne
import pandas as pd

if __name__ == '__main__':
    dir_data = 'Dados/PEAC_Ativo - Análise - Aline e André'
    eeg_ch_names_file = "eeg_channels.csv"
    DROP_CHN = ['33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '68', '69']
    participant = 'Leila Santos'

    mne_eeg_file = "Leila Santos PEAC L ativo.vhdr"
    mne_eeg_resampled_file = "Leila Santos PEAC L ativo.fif"
    csv_eeg_resampled_file = "Leila Santos PEAC L ativo.csv"
    mne_anot_file = "Leila Santos PEAC L ativo.vmrk"
    csv_anot_file = mne_anot_file.replace('.vmrk', ' annotations.csv')

    mne_eeg_path = os.path.join(dir_data, participant, mne_eeg_file)
    mne_eeg_resampled_path = os.path.join(dir_data, participant, 'resampled', mne_eeg_resampled_file)
    csv_eeg_resampled_path = os.path.join(dir_data, participant, 'resampled', csv_eeg_resampled_file)
    mne_anot_path = os.path.join(dir_data, participant, mne_anot_file)
    csv_anot_path = os.path.join(dir_data, participant, csv_anot_file)

    raw = mne.io.read_raw_brainvision(
        mne_eeg_path,
        preload=True,
        verbose=False
    )

    eeg_ch_names = pd.read_csv(eeg_ch_names_file, header=None)

    dict_mapping = {}
    for ind, row in eeg_ch_names.iterrows():
        dict_mapping[str(row[0])] =  row[1]

    del dict_mapping['5']

    fs_resample = 1000

    raw_resamp = raw.resample(fs_resample)

    raw_resamp.drop_channels(DROP_CHN)

    mne.rename_channels(raw_resamp.info, dict_mapping)

    raw_resamp.save(mne_eeg_resampled_path)

    df_all = raw_resamp.to_data_frame(picks=['eeg'])
    df_all.to_csv(csv_eeg_resampled_path, index=False)

    annotations = mne.read_annotations(
        mne_anot_path
    )

    annotations_df = annotations.to_data_frame()

    annotations_df.to_csv(csv_anot_path, index=False)
