## ------------------------------------------ packages ------------------------------------------ ##
import tensorflow as tf
import tensorflow_io as tfio
import sounddevice as sd
from time import time_ns
from scipy.io.wavfile import write
import argparse

## ----------------------------------------- arguments ------------------------------------------ ##
parser = argparse.ArgumentParser()
parser.add_argument("--device", default=0, type=int)
args = parser.parse_args()

## ------------------------------------- initial variables -------------------------------------- ##
DEVICE = args.device
SAMPLING_FREQUENCY = 16000
RESOLUTION = 'int16'
CHANNELS = 1

## -------------------------------------- global variables -------------------------------------- ##
global audio_buffer
# allocate 1s of buffer
audio_buffer = tf.Variable(tf.zeros((SAMPLING_FREQUENCY), dtype=tf.float32))

## ------------------------------------------ classes ------------------------------------------- ##
class AudioReader():
    def __init__(self, resolution, sampling_rate):
        self.resolution = resolution
        self.sampling_rate = sampling_rate

    def get_audio(self, filename):
        audio_io_tensor = tfio.audio.AudioIOTensor(filename, self.resolution)        

        audio_tensor = audio_io_tensor.to_tensor()
        audio_tensor = tf.squeeze(audio_tensor)

        audio_float32 = tf.cast(audio_tensor, tf.float32)
        audio_normalized = audio_float32 / self.resolution.max

        zero_padding = tf.zeros(self.sampling_rate - tf.shape(audio_normalized), dtype=tf.float32)
        audio_padded = tf.concat([audio_normalized, zero_padding], axis=0)

        return audio_padded

    def get_label(self, filename):
        path_parts = tf.strings.split(filename, '/')
        path_end = path_parts[-1]
        file_parts = tf.strings.split(path_end, '_')
        label = file_parts[0]
        
        return label

    def get_audio_and_label(self, filename):
        audio = self.get_audio(filename)
        label = self.get_label(filename)

        return audio, label

class Spectrogram():
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s):
        self.frame_length = int(frame_length_in_s * sampling_rate)
        self.frame_step = int(frame_step_in_s * sampling_rate)

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(
            audio, 
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_spectrogram_and_label(self, audio, label):
        audio = self.get_spectrogram(audio)

        return self.spectrogram, label

class MelSpectrogram():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency
    ):
        self.spectrogram_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        num_spectrogram_bins = self.spectrogram_processor.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sampling_rate,
            lower_edge_hertz=lower_frequency,
            upper_edge_hertz=upper_frequency
        )

    def get_mel_spec(self, audio):
        spectrogram = self.spectrogram_processor.get_spectrogram(audio)
        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        return log_mel_spectrogram

    def get_mel_spec_and_label(self, audio, label):
        log_mel_spectrogram = self.get_mel_spec(audio)

        return log_mel_spectrogram, label


class VAD():
    def __init__(
        self,
        sampling_rate,
        frame_length_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        dbFSthres, 
        duration_thres
    ):
        self.frame_length_in_s = frame_length_in_s
        self.mel_spec_processor = MelSpectrogram(
            sampling_rate, frame_length_in_s, frame_length_in_s, num_mel_bins, lower_frequency, upper_frequency
        )
        self.dbFSthres = dbFSthres
        self.duration_thres = duration_thres

    def is_silence(self, audio):
        log_mel_spec = self.mel_spec_processor.get_mel_spec(audio)
        dbFS = 20 * log_mel_spec
        energy = tf.math.reduce_mean(dbFS, axis=1)

        non_silence = energy > self.dbFSthres
        non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
        non_silence_duration = (non_silence_frames + 1) * self.frame_length_in_s

        if non_silence_duration > self.duration_thres:
            return 0
        else:
            return 1

## ----------------------------------------- functions ------------------------------------------ ##
def callback(indata, frames, callback_time, status):
    """This is called (from a separate thread) for each audio block."""
    global audio_buffer

    # preprocessing audio to tf, squeezing, and normalization by resolution of int16
    tf_indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    tf_indata = tf.squeeze(tf_indata)
    tf_indata = tf_indata/tf.int16.max

    # swap first half of audio buffer with past [-1s, -0.5s] and later half of buffer with [-0.5s, 0s]
    audio_buffer.assign(tf.concat([audio_buffer[SAMPLING_FREQUENCY//2:], tf_indata], axis=0))

    # if it is not silence, it will store the audio
    if not voice_detector.is_silence(audio_buffer):
        print("Voice detected")
        timestamp = int(time_ns()//1e6) # timestamp in ms since epoch
        np_1s_audio = audio_buffer.numpy() # convert to numpy to save wave file
        write(f'.\{timestamp}.wav', SAMPLING_FREQUENCY, np_1s_audio)

## ----------------------------------------- main-loop ------------------------------------------ ##
# voice detector with best parameters from ex1.1
voice_detector = VAD(sampling_rate=SAMPLING_FREQUENCY, frame_length_in_s=0.064, num_mel_bins=16,lower_frequency=60, upper_frequency=500, dbFSthres=-55, duration_thres=0.1)
with sd.InputStream(device=DEVICE, channels=CHANNELS, dtype=RESOLUTION, samplerate=SAMPLING_FREQUENCY, blocksize=SAMPLING_FREQUENCY//2, callback=callback):
    while True:
        key = input()
        if key in ('q', 'Q'):
            print('Stop recording.')
            break