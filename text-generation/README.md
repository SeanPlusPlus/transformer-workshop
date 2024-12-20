# Text Generation


Welcome to the Text Generation module of the transformer-workshop repository! This subdirectory focuses on the exciting world of text generation using pre-trained transformer models. Whether you are interested in creative writing, building chatbots, or experimenting with AI storytelling, this module will introduce you to the foundational concepts and tools for generating coherent and contextually relevant text with just a few lines of code.

---

## Project Setup

Set Python 3.11.6 as the Local Version
 - Navigate to the project directory and set the local version:

```
pyenv local 3.11.6
```
### Creating a Virtual Environment

Create the Virtual Environment
- Use virtualenv to create an isolated environment with Python 3.11.6:

```
virtualenv venv --python=$(pyenv which python)
```

Activate the Virtual Environment

```
source venv/bin/activate
```

### Installing Required Libraries

Install the Python libraries:

```
pip install -r requirements.txt
```

### Running the Project

```
cd text-generation
python hello_world.py
```

You should see output like

```
Generated Text 1: Once upon a time, I have been living peacefully in another world. I now spend my evenings in a strange land, in a strange town a little over fifty miles from their home where we are living. I make my way through the strange village,
```

Alright, time to add some more interesting scripts to our tutorial!

---

### Direct Use of `GPT2LMHeadModel` and `GPT2Tokenizer`

```
python gpt2_intro.py
```

Output

```
*** Once upon a time it must have been an ancient tradition that at the same time the city's rich culture was being threatened by an overzealous bureaucracy. It was a moment of wonderment of how they could make sense of an entire nation and the
```

**Overview**

- This approach gives us direct access to the tokenizer and model, allowing us to configure the text generation process in detail.

- We can modify low-level parameters (like the `no_repeat_ngram_size`, `top_k`, and `top_p`) which allow us to control how diverse or repetitive the generated text is.

- If we need to customize how text is tokenized, or work directly with the raw input/output tensors, this is the best approach.

- Since this method deals with the model directly, we need to handle the data format and tensors explicitly. `input_ids` are encoded in PyTorch tensor format (`return_tensors='pt'`).

- After generating text, we need to decode it manually using the tokenizer.

**When to Use**

- When we need greater control over the text generation process.

- When experimenting with customized model settings and trying out advanced configurations.

- When working directly with PyTorch and needing more fine-grained model manipulation.

**Contrast with the `pipeline` API used in hello world**

- The pipeline function is a high-level abstraction that simplifies common tasks like text generation, question answering, and more.
- Fewer Lines of Code: With pipeline, you don't need to worry about tokenization or decoding explicitly. It takes care of these processes internally, allowing you to write less code.
- Less Flexibility: The trade-off for this simplicity is that you have less control over the specific parameters of the model. While you can still adjust things like max_length and temperature, you don't have direct access to the raw model and tokenizer.
- All-in-One: The pipeline handles everything—from model loading to generation and decoding—which is great for quick prototyping.

When to Use:

- When we want quick, straightforward results without needing in-depth configuration.
- When building prototypes or getting started with text generation.
- When you don’t need detailed control over the generation process or are just trying things out.

### Text Generation Tests

```
python text_gen_tests.py
```

**What This Script Does**

1. Predefined Test Cases:
    - The test_cases list contains dictionaries that define different combinations of prompts and parameters.
    - Each dictionary includes: prompt, max_length, temperature, top_k, top_p, and num_return_sequences.

2. Automated Exploration:
    - Instead of prompting for input, the script iterates over the test_cases list and generates text based on each configuration.
    - This allows you to quickly see the impact of different parameters on the generated output.

This approach is great for testing and understanding how different settings influence the generated text without manual input, allowing you to iterate and analyze efficiently.

### Controlled Text Generation

```python controlled_text_gen.py```


**Overview**

The `controlled_text_gen` script generates text based on predefined prompts using the GPT-2 model. It applies constraints to filter generated text and outputs the results into an HTML file, styled with Bootstrap, for easy review.

This functionality is split into two parts:

1. Core Logic (`controlled_text_gen.py`): Handles text generation and processing.
2. HTML Generation (`html_generator.py`): Encapsulates HTML rendering logic to keep the core script clean and focused.


#### Key Features `controlled_text_gen.py`

1. Model and Tokenizer Setup:
    - Utilizes GPT-2 from Hugging Face's transformers library.
    - Automatically configures the device (cuda or cpu) for efficient execution.

2. Prompt Definition:
    - Generates text for a set of predefined prompts (e.g., poetry, explanations, and descriptions).

3. Custom Constraints:
    - Filters generated text based on banned words (e.g., bad, terrible, worst).

4. Text Output:
    - Encodes filtered results into an HTML structure via a helper function imported from html_generator.py.

5. HTML File Generation:
    - Outputs results to a styled HTML file, which is automatically opened in the default web browser.

#### How It Works

Text Generation Process

1. Prompt Input: Prompts are tokenized using the GPT-2 tokenizer.

2. Model Inference: The GPT-2 model generates text using sampling-based techniques with configurable parameters:
    - `temperature`: Adjusts randomness in text generation.
    - `top_k` and `top_p`: Controls sampling quality.
    - `num_return_sequences`: Specifies how many variations are generated per prompt.

3. Constraint Filtering: Texts containing banned words are flagged and excluded from the output.

4. HTML Rendering:
    - Prompts and generated texts are organized into styled cards using Bootstrap.
    - Skipped texts are highlighted for clarity.

# Fine-Tuning GPT-2 with HTML Output

`gpt2_fine_tuning.py`

## Overview
This script fine-tunes OpenAI's GPT-2 model using a small, predefined dataset and generates text based on custom prompts. It also displays the generated output in an HTML file that opens in your default web browser.

---

## Features

1. **Fine-Tuning**:
   - Fine-tunes GPT-2 with custom training and evaluation datasets.
   - Supports tokenization, padding, and truncation for consistent input lengths.
   - Saves the fine-tuned model for reuse.

2. **Text Generation**:
   - Generates text based on user-defined prompts.
   - Outputs generated text into a styled HTML file for easy visualization.

3. **Integrated HTML Output**:
   - Dynamically creates an HTML file with prompts and their corresponding generated text.
   - Automatically opens the HTML file in your browser.

---

## Script Breakdown

### 1. Model and Tokenizer Setup
- Loads the pre-trained GPT-2 model and tokenizer from Hugging Face.
- Ensures the tokenizer has a padding token for proper sequence alignment.

```python
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

### 2. Dataset Definition
- Defines training and evaluation datasets directly in the script for simplicity.
- Converts datasets into the 🤗 `Dataset` format.

```python
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
```

### 3. Tokenization
- Tokenizes datasets with truncation and padding to ensure consistent input lengths.

```python
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)
```

### 4. Fine-Tuning
- Uses the Hugging Face `Trainer` API for fine-tuning.
- Configures training arguments for epochs, batch size, and learning rate.
- Saves the fine-tuned model.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

### 5. Text Generation
- Loads the fine-tuned model and generates text for predefined prompts.
- Dynamically generates HTML content with the results.

```python
prompts = ["AI is", "The weather", "Fine-tuning GPT"]
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 6. HTML Output
- Creates an HTML file styled with inline CSS to display prompts and their generated outputs.
- Automatically opens the HTML file in the default browser.

```python
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

# Open in browser
webbrowser.open(f"file://{os.path.abspath(html_file)}")
```

---

## How to Run

1. **Install Dependencies**:
   - Install the required Python libraries:
     ```bash
     pip install torch transformers datasets
     ```

2. **Run the Script**:
   - Execute the script to fine-tune the model and generate text:
     ```bash
     python gpt2_fine_tuning.py
     ```

3. **View Output**:
   - The script automatically opens the HTML file in your default web browser.


