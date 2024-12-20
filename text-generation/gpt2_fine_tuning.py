import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
from html import escape
import webbrowser

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure the tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token

# Ensure CUDA is available for faster training
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define training and evaluation datasets directly in the script
train_data = [
    "The sun is shining brightly today.",
    "AI is revolutionizing many industries.",
    "Fine-tuning GPT-2 allows customization for specific tasks.",
]

eval_data = [
    "The weather forecast predicts a rainy weekend.",
    "Natural language processing is a fascinating field.",
    "Large language models can generate coherent text.",
]

# Convert data into datasets using the 🤗 Datasets library
train_dataset = Dataset.from_dict({"text": train_data})
eval_dataset = Dataset.from_dict({"text": eval_data})

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    report_to="none",  # Disable report to prevent errors without proper API keys
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
output_dir = "fine_tuned_gpt2"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine-tuned model saved to {output_dir}")

# Load the fine-tuned model for generation
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Generate and display results in HTML
prompts = ["AI is", "The weather", "Fine-tuning GPT"]
output_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Generated Text</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        .prompt {
            font-weight: bold;
            color: #2c3e50;
        }
        .output {
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #ddd;
            background: #f9f9f9;
        }
    </style>
</head>
<body>
    <h1>Generated Text Output</h1>
"""

for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    output_html += f"""
    <div class='output'>
        <p class='prompt'>Prompt: {escape(prompt)}</p>
        <p>Generated Text: {escape(generated_text)}</p>
    </div>
    """

output_html += """
</body>
</html>
"""

# Write to an HTML file
html_file = "generated_text.html"
with open(html_file, "w", encoding="utf-8") as f:
    f.write(output_html)

# Open in the browser
webbrowser.open(f"file://{os.path.abspath(html_file)}")
