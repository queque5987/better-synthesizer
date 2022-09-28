from pathlib import Path
from synthesizer import inference as synth

def inference(embed, text):
    synthesizer = synth.Synthesizer("/")
    texts = [text]
    embeds = [embed]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    return spec
