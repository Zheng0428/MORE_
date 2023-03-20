from transformers import GPT2Config
import torch
from src.tasks.gpt2_CVAE import Decoder
from src.tasks.gpt2 import GPT2LMHeadModel

lxrt_feature = torch.rand(32, 100, 768)
rtg = torch.rand(32,100)
text = 'i love HUHU'
timesteps = torch.arange(start=0, end=100, step=1).unsqueeze(0)
timesteps = torch.repeat_interleave(timesteps, 32, dim = 0) 
traj_mask = torch.randint(low=0, high=2, size=(32, 100))
config = GPT2Config()
# more_decoder = Decoder(config, add_input=True, add_attn=True, attn_proj_vary=True)
more_decoder = GPT2LMHeadModel.from_pretrained('gpt2', num_hidden_layers = 6)
for i, layer in enumerate(more_decoder.transformer.h):
    if i >= more_decoder.transformer.config.n_layer - 4:
        for param in layer.parameters():
            param.requires_grad = False

# tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2") #load tokenizer
# tokenizer.pad_token = tokenizer.eos_token
texts = [
    'This is the first text.',
    'This is the second text.',
    'This is the third text.',
    'This is the fourth text.',
    'This is the fifth text.',
    'This is the sixth text.',
    'This is the seventh text.',
    'This is the eighth text.',
    'This is the ninth text.',
    'This is the tenth text.',
    'This is the eleventh text.',
    'This is the twelfth text.',
    'This is the thirteenth text.',
    'This is the fourteenth text.',
    'This is the fifteenth text.',
    'This is the sixteenth text.',
    'This is the seventeenth text.',
    'This is the eighteenth text.',
    'This is the nineteenth text.',
    'This is the twentieth text.',
    'This is the first text.',
    'This is the second text.',
    'This is the third text.',
    'This is the fourth text.',
    'This is the fifth text.',
    'This is the sixth text.',
    'This is the seventh text.',
    'This is the eighth text.',
    'This is the ninth text.',
    'This is the tenth text.',
    'This is the eleventh text.',
    'This is the twelfth text.',
]
        
#target = tokenizer.batch_encode_plus(texts, padding = 'max_length', truncation = True, return_tensors = 'pt', max_length = 20, add_special_tokens = True, return_attention_mask = True, return_token_type_ids = False)
output = more_decoder(inputs_embeds = lxrt_feature, labels = lxrt_feature, attention_mask = traj_mask, position_ids = timesteps, rtg = rtg)        #Be sure to pay attention to whether the input sequences are of the same length  #past_key_values = past 后面有时间可以加上
loss = output.loss
c = 1