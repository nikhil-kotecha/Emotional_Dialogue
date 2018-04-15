import pyaudio
import wave
import sys

def record(format=pyaudio.paInt16, channels=1, rate=16000, chunk=1024, record_seconds=5):
    p = pyaudio.PyAudio()
    stream = p.open(formmat, channels, rate, input=True, frames_per_buffer=chunk)

    frames = []

    for i in range(0, rate // (chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    return frames



