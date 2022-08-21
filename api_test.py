import requests
import json
import librosa
import numpy as np

import soundfile as sf

def get_embed():
    # text = "Excelent! It's a rocket. this is the good example of your voice try again."
    # url = "http://localhost/encoder/inference/"
    url = "https://better-encoder.herokuapp.com/inference/"

    wav, sr = librosa.load("19-227-0009.wav")
    print(sr)
    wav = wav.tolist()
    # print(type(sr))
    wav_json = json.dumps({
        "wav": wav,
        "sr": sr
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("GET", url, headers=headers, data=wav_json)
    # print(response.text)
    # temp = ''
    # wav = []
    print(type(response.json()))
    embed = np.array(response.json())
    return embed
    # spec = synthesizer.inference(embed, "I love you.")
    # print(spec)
    # for dat in response.text:
    #     if dat == "[": continue
    #     if dat == "," or dat == "]":
    #         wav.append(float(temp))
    #         temp = ''
    #         continue
    #     temp += dat
    # print(wav)
    # wav = response.json()
    # sr = wav[-1]
    # wav = np.array(wav[:-1])
    # sf.write("testing_internal_server_man.wav", wav.astype(np.float32), 16000)
    # # sf.write("testing_internal_server_woman.wav", wav.astype(np.float32), 22050)