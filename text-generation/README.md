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