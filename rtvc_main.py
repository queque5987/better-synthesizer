import librosa
from pathlib import Path

from synthesizer.inference import Synthesizer
# import api_test

class rtvc_args():
    def __init__(self):
        self.syn_model_fpath = Path("saved_models/synthesizer")
        self.cpu = True
        self.seed = None
    def pop(self, idx):
        if idx == "cpu":
            return self.cpu

def inference(embed, text):
    args = rtvc_args()
    synthesizer = Synthesizer(args.syn_model_fpath)
    synthesizer.model_state_load()
    texts = [text]
    embeds = [embed]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    # print(type(spec))
    return spec

# if __name__ == "__main__":
#     embed = api_test.get_embed()
#     spec = inference(embed, "I love you")
#     print(spec)
    # args = rtvc_args()
    # synthesizer = Synthesizer(args.syn_model_fpath)
    # text = ""
    # embed = []
    # # The synthesizer works in batch, so you need to put your data in a list or numpy array
    # texts = [text]
    # embeds = [embed]
    # # If you know what the attention layer alignments are, you can retrieve them here by
    # # passing return_alignments=True
    # specs = synthesizer.synthesize_spectrograms(texts, embeds)
    # spec = specs[0]
    # print("Created the mel spectrogram")
    # ## Generating the waveform
    # print("Synthesizing the waveform:")