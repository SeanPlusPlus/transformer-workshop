# Transformers

High-Level Hierarchy of Interacting with Transformers


| Approach                 | Ease of Use | Flexibility | Cost       | Computational Requirements | When to Use                                           |
|--------------------------|-------------|-------------|------------|----------------------------|------------------------------------------------------|
| 1. OpenAI API               | High        | Low         | Pay-as-you-go | Low                        | Quick results, no setup required                    |
| 2. Pre-Trained Model        | Medium      | Medium      | Free        | Medium                     | Use cases align with pre-trained capabilities        |
| 3. Fine-Tuning              | Low         | High        | Medium      | High                       | Task-specific customization with labeled data       |
| 4. Building from Scratch    | Very Low    | Very High   | High        | Very High                  | For research or custom architectures                |


---

### 1. Hitting the OpenAI API (Out-of-the-Box Solution)

Use a hosted service like OpenAI’s API to access models such as GPT-4 or DALL·E.
    
- Pros:
    - No infrastructure or training required.
    - Instant access to state-of-the-art models.
    - Scales automatically with minimal effort.

- Cons:
    - Limited flexibility (can’t customize the model directly).
    - Costs can add up for high-volume usage.
    - Dependency on external providers.
    
#### Examples for Hitting the OpenAI API

- Text Generation (Creative Writing)

- Customer Support (Chatbots)

- Text Summarization

- Code Generation

- All the fun hacky games I build that leverage the API

---

### 2. Using a Pre-Trained Model (Local Inference)

Load a pre-trained model locally using libraries like Hugging Face Transformers and run inference on your data.
    
- Pros:
    - No training required.
    - Full control over deployment infrastructure.
    - Often free (just download the model).

- Cons:
    - Limited to what pre-trained model was used for.
    - Requires setting up local env and dependencies

#### Examples

**Sentiment Analysis**

Classifying input text as positive, negative, or neutral. 

```python
from transformers import pipeline

# Load pre-trained sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Input text
text = "I absolutely loved this movie! It was fantastic."

# Get predictions
result = sentiment_analyzer(text)
print(result)

# Output
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

**Named Entity Recognition (NER)**

Identifies and categorizes entities in text, such as names, organizations, and locations.


```python
from transformers import pipeline

# Load pre-trained NER pipeline
ner_pipeline = pipeline("ner", grouped_entities=True)

# Input text
text = "Barack Obama was born in Hawaii and became the 44th President of the United States."

# Get entities
result = ner_pipeline(text)
print(result)

# Output
# [{'entity_group': 'PER', 'score': 0.998, 'word': 'Barack Obama'},
#  {'entity_group': 'LOC', 'score': 0.999, 'word': 'Hawaii'},
#  {'entity_group': 'ORG', 'score': 0.995, 'word': 'United States'}]
```

**Text Summarization**

Condenses long articles or passages into shorter summaries.

```python
from transformers import pipeline

# Load pre-trained summarization pipeline
summarizer = pipeline("summarization")

# Input text
text = """
Artificial Intelligence (AI) is transforming the world. From healthcare to transportation,
AI is being used to improve efficiency, reduce costs, and solve complex problems.
However, challenges such as ethical considerations and data privacy remain critical.
"""

# Get summary
result = summarizer(text, max_length=50, min_length=10, do_sample=False)
print(result)

# Output
# [{'summary_text': 'Artificial Intelligence is transforming industries like healthcare and transportation,
# improving efficiency and reducing costs. Challenges include ethical considerations and data privacy.'}]
```

**Question Answering**

Answer a question.

```python
from transformers import pipeline

# Load pre-trained question-answering pipeline
qa_pipeline = pipeline("question-answering")

# Input question and context
context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
"""
question = "Who designed the Eiffel Tower?"

# Get answer
result = qa_pipeline(question=question, context=context)
print(result)

# Output
# {'score': 0.99, 'start': 63, 'end': 78, 'answer': 'Gustave Eiffel'}
```

Why These Examples Work

1. Sentiment Analysis demonstrates a common and easy-to-visualize use case.
2. NER showcases the ability to extract structured information from unstructured text.
3. Text Summarization is valuable for condensing large volumes of information.
4. Question Answering highlights the model’s ability to understand and reason based on context.

These tasks are straightforward to implement and provide practical utility, making them perfect for showcasing pre-trained models in action. 

Using a pre-trained model locally provides greater control over the inference process, allowing you to customize aspects like tokenization, batch sizes, and runtime behavior. Additionally, it avoids reliance on external APIs, reducing latency, costs, and potential privacy concerns since all computations happen on your own infrastructure.

---

### 3. Fine-Tuning a Pre-Trained Model

Customize a pre-trained model for your specific task or dataset by fine-tuning its weights.
    
- Pros:
    - Combines the benefits of pre-training with task-specific performance.
    - More flexibility to adapt the model to niche use cases.

- Cons:
    - Requires labeled data and computational resources for training.
    - Takes time to fine-tune and evaluate the model.

#### Examples

**1. Sentiment Analysis on a Custom Dataset**: Fine-tune a model like `distilbert-base-uncased` for sentiment analysis on a specific dataset, such as product reviews or tweets. General-purpose sentiment analysis models may not work well on niche datasets. Fine-tuning ensures the model understands domain-specific vocabulary and context.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load dataset
dataset = load_dataset("your_custom_dataset")

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess(data):
    return tokenizer(data["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(preprocess, batched=True)

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

# Fine-tune model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()
```

**2. Named Entity Recognition (NER) for Domain-Specific Terms**: Fine-tune a model like `bert-base-cased` for identifying entities specific to a domain, such as medical terms or legal jargon. General NER models may not recognize entities in specialized fields without adaptation.

Example:
- Dataset: Sentences annotated with terms like DISEASE, DRUG, and SYMPTOM.
- Outcome: A model that can extract entities from medical records or research papers.

**3. Question Answering on a Specific Dataset**: Fine-tune a pre-trained model like `bert-large-uncased-whole-word-masking-finetuned-squad` for answering questions in a niche dataset. Domain-specific datasets often include unique phrasing and context not covered by general-purpose models.

Example:
- Dataset: A custom FAQ or knowledge base (e.g., a company's internal wiki).
- Outcome: A question-answering model tailored to your dataset for improved accuracy.

**4. Text Classification for Multilabel Tasks**: Fine-tune a model like `roberta-base` for classifying text into multiple categories simultaneously. Pre-trained models may not support multilabel classification without fine-tuning.

Example:
- Dataset: News articles tagged with multiple topics (e.g., politics, sports, technology).
- Outcome: A model capable of assigning multiple tags to a single article.

Why Fine-Tuning is Powerful
- Fine-tuning allows pre-trained models to specialize in domain-specific tasks, outperforming general-purpose APIs or models.
- It provides flexibility for tasks that don’t fit into the out-of-the-box pipelines or pre-trained functionality.


---

### 4. Building a Model from Scratch:

Building a model from scratch is a rare and resource-intensive endeavor, but it is appropriate in specific scenarios where pre-trained models or fine-tuning are insufficient.

- Pros:
    - Total control over architecture and training process.
    - Ability to innovate beyond existing pre-trained architectures.

- Cons:
    - Requires vast computational resources and expertise.
    - Needs massive datasets for pre-training to achieve competitive performance.

#### Examples for Building a Model from Scratch

1. Researching Novel Architectures
    - What it is: Designing and testing a new transformer architecture for academic or experimental purposes.
    - Why It’s Appropriate:
        - When pushing the boundaries of current AI capabilities.
        - Exploring innovations in model efficiency, interpretability, or scalability.
    - Example Use Case:
        - Developing a transformer model optimized for edge devices with significantly reduced computational requirements.
        - Researching a multi-modal transformer that simultaneously processes text, images, and audio.

2. Addressing a Highly Specialized Domain
    - What it is: Creating a model tailored for a domain where no suitable pre-trained models or datasets exist.
    - Why It’s Appropriate:
        - When the domain-specific vocabulary, structure, or data distribution is fundamentally different from those used in existing pre-trained models.
    - Example Use Case:
        - Building a model for highly technical fields like quantum physics, which requires understanding specialized equations and terms.
        - Training a transformer on ancient scripts or languages with no modern datasets (e.g., Sumerian or Mayan hieroglyphs).

3. Training a Multilingual Model for Low-Resource Languages
    - What it is: Building a language model for underrepresented or endangered languages.
    - Why It’s Appropriate:
        - Pre-trained models often perform poorly on low-resource languages due to insufficient training data.
        - A custom-built model can incorporate linguistic nuances and structure specific to those languages.
    - Example Use Case:
        - Developing a transformer trained on multiple dialects of an endangered language to support language preservation and revitalization.
        - Training a translation model for languages with no direct datasets (e.g., translating between small, local languages in a remote region).

**Why Building From Scratch is Rare**

1. Resource-Intensive: Requires massive datasets and significant computational resources (e.g., TPUs or GPUs running for weeks).
2. Expertise Required: Demands deep knowledge of transformer architectures, optimization techniques, and large-scale data processing.
3. Existing Models Are Often Enough: Pre-trained or fine-tuned models usually cover most real-world needs efficiently.

**Key Takeaway**

Building a model from scratch is best reserved for cutting-edge research or ultra-specialized domains where existing models cannot be adapted or extended. For most practical applications, fine-tuning or using pre-trained models will suffice.

---
