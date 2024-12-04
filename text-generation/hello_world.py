from transformers import pipeline

# Load a pre-trained text generation model
generator = pipeline('text-generation', model='gpt2')

# Generate text based on a simple prompt
prompt = "Once upon a time"
result = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
for i, generated_text in enumerate(result):
    print(f"Generated Text {i + 1}: {generated_text['generated_text']}")
