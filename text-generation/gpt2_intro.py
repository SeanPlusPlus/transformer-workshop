import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Input prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate text
generated_outputs = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0
)

# Decode and print the output
generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
print(generated_text)

