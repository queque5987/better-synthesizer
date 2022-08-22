import requests
import json
import torch

url = "https://better-synthesizer-w.herokuapp.com/generate/"

def generate(chars, batched_embeds):
    chars = [c.tolist() for c in chars]
    batched_embeds = [b.tolist() for b in batched_embeds]
    wav_json = json.dumps({
        "chars": chars,
        "batched_embeds": batched_embeds
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("GET", url, headers=headers, data=wav_json)
    mel = response.json()
    mel = torch.tensor(mel)
    print(mel)

    return None, mel, None