import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

# 初始化 tokenizer 和模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 输入文本列表
input_text_list = ["Hello, how are you doing today?", "What's the weather like tomorrow?"]
tokenizer.pad_token = tokenizer.eos_token
# 将文本编码为 tokens
input_ids = tokenizer.batch_encode_plus(
    input_text_list, 
    padding = 'max_length', 
    truncation = True, 
    return_tensors = 'pt', 
    max_length = 20, 
    add_special_tokens = True, 
    return_attention_mask = True, 
    return_token_type_ids = False)
input_ids = input_ids.data['input_ids']

embedding_outputs = model.transformer.wte(input_ids)

# Print the shape of the embedding outputs
print(embedding_outputs.shape)
# 生成 embeddings


