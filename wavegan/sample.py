import logging
import librosa
import pescador
import os
import numpy as np
import pdb


LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)

class_labels = {
    'airport': 0,
    'bus': 1,
    'metro': 2,
    'metro_station': 3,
    'park': 4,
    'public_square': 5,
    'shopping_mall': 6,
    'street_pedestrian': 7,
    'street_traffic': 8,
    'tram': 9
}


def file_sample_generator(meta, window_length=65536, fs=16000):
    """
    Audio sample generator
    """
    filepath, label = meta
    try:
        audio_data, _ = librosa.load(filepath, sr=fs)

        # Clip amplitude
        max_amp = np.max(np.abs(audio_data))
        if max_amp > 1:
            audio_data /= max_amp
    except Exception as e:
        LOGGER.error('Could not load {}: {}'.format(filepath, str(e)))
        raise StopIteration()

    audio_len = len(audio_data)

    # Pad audio to at least a single frame
    if audio_len < window_length:
        pad_length = window_length - audio_len
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad

        audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')
        audio_len = len(audio_data)

    while True:
        if audio_len == window_length:
            # If we only have a single frame's worth of audio, just yield the whole audio
            sample = audio_data
        else:
            # Sample a random window from the audio file
            start_idx = np.random.randint(0, audio_len - window_length)
            end_idx = start_idx + window_length
            sample = audio_data[start_idx:end_idx]

        sample = sample.astype('float32')
        assert not np.any(np.isnan(sample))

        if label is None:
            yield {'X': sample}
        else:
            yield {'X': sample, 'Y': label}


def create_batch_generator(audio_filepath_list, batch_size):
    streamers = []
    for audio_filepath in audio_filepath_list:
        class_name = os.path.basename(audio_filepath).split('-')[0]
        if class_name in class_labels:
            label = class_labels[class_name]
        else:
            label = None
        s = pescador.Streamer(file_sample_generator, [audio_filepath, label])
        streamers.append(s)

    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)

    return batch_gen


def get_all_audio_filepaths(audio_dir):
    return [os.path.join(root, fname)
            for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names
            if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]


def create_data_split(audio_filepath_list, valid_ratio, test_ratio,
                      train_batch_size, valid_size, test_size):
    num_files = len(audio_filepath_list)
    num_valid = int(np.ceil(num_files * valid_ratio))
    num_test = int(np.ceil(num_files * test_ratio))
    num_train = num_files - num_valid - num_test

    assert num_valid > 0
    assert num_test > 0
    assert num_train > 0

    valid_files = audio_filepath_list[:num_valid]
    test_files = audio_filepath_list[num_valid:num_valid + num_test]
    train_files = audio_filepath_list[num_valid + num_test:]

    train_gen = create_batch_generator(train_files, train_batch_size)
    valid_data = next(iter(create_batch_generator(valid_files, valid_size)))
    test_data = next(iter(create_batch_generator(test_files, test_size)))

    return train_gen, valid_data, test_data
