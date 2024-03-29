## ------------------------------------------ packages ------------------------------------------ ##
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_model_optimization as tfmot
import sounddevice as sd
from scipy.io.wavfile import write
import argparse
import psutil
import uuid
import time
from datetime import datetime
import redis
import os
import shutil
import zipfile


## ----------------------------------------- arguments ------------------------------------------ ##
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--host', default='redis-15734.c293.eu-central-1-1.ec2.cloud.redislabs.com', type=str)
parser.add_argument('--port', default=15734, type=int)
parser.add_argument('--user', default='default', type=str)
parser.add_argument('--password', default='L91dOJlgRLTtJ2zc0cYJbc5pOvp5Vbrg', type=str)
args = parser.parse_args()


## ------------------------------------- initial variables -------------------------------------- ##
DEVICE = args.device
REDIS_HOST = args.host
REDIS_PORT = args.port
REDIS_USERNAME = args.user
REDIS_PASSWORD = args.password

MODEL_NAME = 'model1'
MODEL_FOLDER = '.'

MAC_ADDRESS = hex(uuid.getnode())
BATTERY_TS_NAME = f'{MAC_ADDRESS}:battery'
POWER_TS_NAME = f'{MAC_ADDRESS}:power'

PREPROCESSING_ARGS = {
    'sampling_rate': 16000,
    'frame_length_in_s': 0.064,
    'frame_step_in_s': 0.016,
    'num_mel_bins': 10,
    'num_coefficients':13,
    'lower_frequency': 0,
    'upper_frequency': 4000
}
RESOLUTION = 'int16'
CHANNELS = 1
LABELS = ['yes', 'no']

## -------------------------------------- global variables -------------------------------------- ##
global audio_buffer
global do_monitoring
audio_buffer = tf.Variable(tf.zeros((PREPROCESSING_ARGS['sampling_rate']), dtype=tf.float32))
do_monitoring = False


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
        spectrogram = self.get_spectrogram(audio)

        return spectrogram, label


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


class MFCC():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        num_coefficients
    ):

        self.mel_spectrogram_processor = MelSpectrogram(
            sampling_rate = sampling_rate, 
            frame_length_in_s = frame_length_in_s, 
            frame_step_in_s = frame_step_in_s, 
            num_mel_bins = num_mel_bins, 
            lower_frequency = lower_frequency, 
            upper_frequency = upper_frequency
        )
        self.num_coefficients = num_coefficients

    def get_mfccs(self, audio):
        log_mel_spectrogram = self.mel_spectrogram_processor.get_mel_spec(audio)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :self.num_coefficients]

        return mfccs

    def get_mfccs_and_label(self, audio, label):
        mfccs = self.get_mfccs(audio)
        return mfccs, label


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


## ------------------------------------- runtime variables -------------------------------------- ##
# voice detector with best parameters from hw1_ex1.1
voice_detector = VAD(sampling_rate=PREPROCESSING_ARGS['sampling_rate'], frame_length_in_s=0.064, num_mel_bins=8, lower_frequency=20, upper_frequency=300, dbFSthres=-20, duration_thres=0.2)

# mfcc processor with best parameters from hw2_ex1.1    
mfcc_processor = MFCC(**PREPROCESSING_ARGS)

# tflite model with best parameters from hw2_ex1.1
if not os.path.exists(f'{MODEL_FOLDER}/{MODEL_NAME}.tflite'):
    with zipfile.ZipFile(f'{MODEL_FOLDER}/{MODEL_NAME}.tflite.zip', 'r') as zipFile:
        zipFile.extractall(f'{MODEL_FOLDER}/model1_temp')
    for root, _, files in os.walk(f'{MODEL_FOLDER}/model1_temp'):
        for currentFile in files:
            fileName, fileExtension = os.path.splitext(currentFile)
            if fileExtension == '.tflite':
                shutil.copy(os.path.join(root, currentFile), f'{MODEL_FOLDER}/{MODEL_NAME}.tflite')
                shutil.rmtree(f'{MODEL_FOLDER}/model1_temp')
interpreter = tf.lite.Interpreter(f'{MODEL_FOLDER}/{MODEL_NAME}.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

## ----------------------------------------- functions ------------------------------------------ ##
def do_classification():
    global audio_buffer
    global do_monitoring

    # preprocessing
    np_1s_audio = audio_buffer.numpy() # convert to numpy to classify wave file
    mfccs = mfcc_processor.get_mfccs(np_1s_audio)
    mfccs = tf.expand_dims(mfccs, 0)
    mfccs = tf.expand_dims(mfccs, -1)
    
    # classification
    interpreter.set_tensor(input_details[0]['index'], mfccs)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_probability = output[0][0]
    print('Predicted Probability of "No":', predicted_probability)
    print('Predicted Probability of "Yes":', 1 - predicted_probability)

    # deciding about the monitoring
    if predicted_probability > 0.99 and do_monitoring:
        do_monitoring = False
    if predicted_probability < 0.01 and not do_monitoring:
        do_monitoring = True
    print('-- Monitoring Activation Status: ', do_monitoring)


def callback(indata, frames, callback_time, status):
    global audio_buffer

    # preprocessing audio to tf, squeezing, and normalization by resolution of int16
    tf_indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    tf_indata = tf.squeeze(tf_indata)
    tf_indata = tf_indata/tf.int16.max

    # swap first half of audio buffer with past [-1s, -0.5s] and later half of buffer with [-0.5s, 0s]
    audio_buffer = tf.concat([audio_buffer[PREPROCESSING_ARGS['sampling_rate']//2:], tf_indata], axis=0)

    # if it is not silence, it will classify the audio
    is_audio_silent = voice_detector.is_silence(audio_buffer)
    if not is_audio_silent:
        print()
        print(f'Voice detected on', datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
        do_classification()


## -------------------------------------- redis connection -------------------------------------- ##
try:   
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME, password=REDIS_PASSWORD)
    is_connected = redis_client.ping()
    print(f'Redis Connection Status: {is_connected}')
except:
    print(f'Redis Connection Failed!')
    exit()

try:
    redis_client.ts().create(BATTERY_TS_NAME)
    redis_client.ts().create(POWER_TS_NAME)
    print('Time-Series Created!')
except:
    print('Time-Series Already Exist!')


## ----------------------------------------- main-loop ------------------------------------------ ##
old_timestamp = time.time()
with sd.InputStream(device=DEVICE, channels=CHANNELS, dtype=RESOLUTION, samplerate=PREPROCESSING_ARGS['sampling_rate'], blocksize=PREPROCESSING_ARGS['sampling_rate']//2, callback=callback):
    
    # infinite loop for voice recording
    while True:

        # infinite loop for monitoring
        while do_monitoring:

            # setting up the parameters
            timestamp = time.time()
            timestamp_ms = int(timestamp * 1000)
            battery_level = psutil.sensors_battery().percent
            power_plugged = int(psutil.sensors_battery().power_plugged)

            # uploading data into redis
            redis_client.ts().add(BATTERY_TS_NAME, timestamp_ms, battery_level)
            redis_client.ts().add(POWER_TS_NAME, timestamp_ms, power_plugged)

            # waiting for 1 second
            time.sleep(1)