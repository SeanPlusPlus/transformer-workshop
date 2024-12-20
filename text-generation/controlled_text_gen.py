import os
import webbrowser
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from html_generator import generate_html_content

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define prompts and constraints
prompts = [
    "Write a poem about the stars:\n",
    "Explain quantum mechanics in simple terms:\n",
    "Describe a futuristic city in vivid detail:\n",
]
custom_ban_words = ["bad", "terrible", "worst"]  # Words to avoid in generation

# Generate text for each prompt and collect outputs
outputs_by_prompt = []
for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=3,
        no_repeat_ngram_size=2,
    )
    decoded_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    outputs_by_prompt.append(decoded_texts)

# Generate HTML content using the imported function
output_html = generate_html_content(prompts, outputs_by_prompt, custom_ban_words)

# Write the output to an HTML file
output_file = "generated_text.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_html)

# Open the HTML file in the default web browser
webbrowser.open("file://" + os.path.abspath(output_file))
