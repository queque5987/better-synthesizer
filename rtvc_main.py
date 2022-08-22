from pathlib import Path
from synthesizer import inference as synth

class rtvc_args():
    def __init__(self):
        self.syn_model_fpath = Path("saved_models/synthesizer.pt")
        self.cpu = True
        self.seed = None
    def pop(self, idx):
        if idx == "cpu":
            return self.cpu

def inference(embed, text):
    args = rtvc_args()
    synthesizer = synth.Synthesizer(args.syn_model_fpath)
    texts = [text]
    embeds = [embed]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    return spec
