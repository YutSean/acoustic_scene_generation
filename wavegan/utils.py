import os
import librosa


def save_samples(epoch_samples, epoch, output_dir, fs=16000):
    """
    Save output samples to disk
    """
    sample_dir = os.path.join(output_dir, str(epoch))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for idx, samp in enumerate(epoch_samples):
        output_path = os.path.join(sample_dir, "{}.wav".format(idx+1))
        samp = samp[0]
        librosa.output.write_wav(output_path, samp, fs)


def save_generated(data, output_dir, start_idx, fs=16000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for offset, sample in enumerate(data):
        output_path = os.path.join(output_dir, "{}.wav".format(start_idx + offset + 1))
        sample = sample[0]
        librosa.output.write_wav(output_path, sample, fs)
