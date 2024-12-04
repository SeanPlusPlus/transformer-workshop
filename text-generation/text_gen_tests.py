import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Predefined set of prompts and parameters
test_cases = [
    {
        "prompt": "The future of AI is",
        "max_length": 50,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 2,
    },
    {
        "prompt": "Once upon a time in a faraway land",
        "max_length": 60,
        "temperature": 1.0,
        "top_k": 0,  # top_k set to 0 means no restrictions
        "top_p": 0.9,
        "num_return_sequences": 3,
    },
    {
        "prompt": "In a surprising turn of events,",
        "max_length": 40,
        "temperature": 1.5,
        "top_k": 30,
        "top_p": 0.85,
        "num_return_sequences": 1,
    },
]

def generate_text(prompt, max_length, temperature, top_k, top_p, num_return_sequences):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    generated_outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )

    # Decode and print the output
    for i, output in enumerate(generated_outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\nPrompt: {prompt}\nGenerated Text {i + 1}:\n{generated_text}\n{'-'*40}")

if __name__ == "__main__":
    # Iterate over predefined test cases
    for test_case in test_cases:
        generate_text(
            prompt=test_case["prompt"],
            max_length=test_case["max_length"],
            temperature=test_case["temperature"],
            top_k=test_case["top_k"],
            top_p=test_case["top_p"],
            num_return_sequences=test_case["num_return_sequences"]
        )
