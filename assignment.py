# ========================================
# 🧠 Named Entity Recognition with spaCy
# ========================================

# Install spaCy and download a model
!pip install -U spacy

import spacy

# Load a small English model
nlp = spacy.load("en_core_web_sm")

# ----------------------------------------
# 1️⃣ Test on a sample text
# ----------------------------------------
text = "Apple is looking to buy a UK-based startup for $1 billion by January 2025."

# Process text
doc = nlp(text)

# Extract named entities
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

# Visualize entities (in Jupyter/Colab only)
from spacy import displacy

displacy.render(doc, style="ent")

# ----------------------------------------
# 2️⃣ Add custom examples and update model
# ----------------------------------------
# Suppose you have domain-specific sentences:
# Let's teach the model a new entity type: 'TECH_COMPANY'

import random
from spacy.training.example import Example

# Create training data
TRAIN_DATA = [
    ("OpenAI released GPT-4.", {"entities": [(0, 6, "TECH_COMPANY")]}),
    ("Hugging Face provides transformer models.", {"entities": [(0, 12, "TECH_COMPANY")]}),
]

# Disable other pipeline components to train only NER
ner = nlp.get_pipe("ner")

# Add new label
ner.add_label("TECH_COMPANY")

# Start training
optimizer = nlp.resume_training()

for i in range(10):
    random.shuffle(TRAIN_DATA)
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer)

# Test updated model
test_text = "Anthropic is another AI research company."
doc = nlp(test_text)
print("Updated Entities:", [(ent.text, ent.label_) for ent in doc.ents])
displacy.render(doc, style="ent")

# ----------------------------------------
# 3️⃣ Save updated model
# ----------------------------------------
nlp.to_disk("./custom_ner_model")

print("✅ Model trained and saved!")
