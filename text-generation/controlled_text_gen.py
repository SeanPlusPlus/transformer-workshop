import os
import webbrowser
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from html import escape

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

# Function to enforce constraints
def is_word_allowed(output_text, banned_words):
    for word in banned_words:
        if word in output_text:
            return False
    return True

# Generate text for each prompt
output_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Generated Text</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: Arial, sans-serif; /* Fallback font style */
            background-color: #f8f9fa; /* Bootstrap light gray */
        }
    </style>
</head>
<body class="bg-light text-dark">
    <div class="container py-5">
        <h1 class="mb-4 text-center">Text Generation Output</h1>
"""

for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate multiple sequences with sampling enabled
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

    output_html += f"""
        <div class="card mb-4 shadow-sm">
            <div class="card-body">
                <h5 class="card-title">Prompt:</h5>
                <p class="card-text"><strong>{escape(prompt.strip())}</strong></p>
                <h6>Generated Texts:</h6>
                <ul class="list-group list-group-flush">
    """
    for i, output in enumerate(outputs):
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)

        # Apply custom constraints
        if is_word_allowed(decoded_text, custom_ban_words):
            output_html += f"""
                <li class="list-group-item">
                    <strong>Generated Text {i+1}:</strong> {escape(decoded_text)}
                </li>
            """
        else:
            output_html += f"""
                <li class="list-group-item text-danger">
                    <strong>Generated Text {i+1}:</strong> [Skipped due to banned words]
                </li>
            """
    output_html += """
                </ul>
            </div>
        </div>
    """

output_html += """
    </div>
</body>
</html>
"""

# Write the output to an HTML file
output_file = "generated_text.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_html)

# Open the HTML file in the default web browser
webbrowser.open("file://" + os.path.abspath(output_file))
