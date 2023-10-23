import os
import uuid
import scipy
import torch
from django.views import View
from django.shortcuts import render
from django.http import JsonResponse
from diffusers import AudioLDMPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if torch.cuda.is_available():
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en",
                                                src_lang="vi_VN",
                                                use_fast=False)
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")
model_vi2en.to(device)

repo_id = "home/namdp/models"
artist = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
artist = artist.to(device)


def translate_vi2en(vi_text):
    input_ids = tokenizer_vi2en(vi_text, return_tensors="pt").input_ids.to(device)
    output_ids = model_vi2en.generate(
        input_ids,
        do_sample=True,
        top_k=20,
        top_p=0.8,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
    )
    en_text = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    en_text = " ".join(en_text)
    return en_text


# Create your views here.
class IndexView(View):
    template_name = 'index.html'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.translator = model_vi2en
        self.tokenizer = tokenizer_vi2en
        self.artist = artist

    def get(self, request):
        context = {
            'image_url': os.path.join('/static', 'proptit_d24.png'),
        }
        return render(request, self.template_name, context=context)

    def post(self, request):
        desc = request.POST['description']

        vi_text = desc.lower()
        en_text = translate_vi2en(vi_text)

        audio = self.artist(en_text, num_inference_steps=10, audio_length_in_s=10.0).audios[0]
        audio_path = os.path.join('./audio', "output.wav")
        scipy.io.wavfile.write(audio_path, rate=16000, data=audio)

        data = {
            'en_text': en_text,
            'vi_text': vi_text,
            'audio_url': audio_path
        }
        return JsonResponse(data)
