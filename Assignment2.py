# Complete Machine Learning Pipeline
# Task 1: Classical ML with Scikit-learn
# Task 2: Deep Learning with TensorFlow
# Task 3: NLP with spaCy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers, models
import spacy
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("🚀 Machine Learning Pipeline Demo")
print("=" * 50)

# ================================================================
# TASK 1: CLASSICAL ML WITH SCIKIT-LEARN
# ================================================================

print("\n✅ TASK 1: Classical ML with Scikit-learn")
print("-" * 40)

# 1. Load Dataset
print("📊 Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")

# Create DataFrame for better visualization
df_iris = pd.DataFrame(X, columns=feature_names)
df_iris['species'] = y
print(f"\nDataset info:")
print(df_iris.head())
print(f"\nClass distribution:")
print(df_iris['species'].value_counts())

# 2. Handle Missing Values (demonstration - Iris has no missing values)
print("\n🔧 Checking for missing values...")
missing_count = pd.DataFrame(X, columns=feature_names).isnull().sum()
print(f"Missing values per feature:\n{missing_count}")

# Create imputer for demonstration (even though not needed)
imputer = SimpleImputer(strategy='mean')

# 3. Encode Labels (already numeric, but demonstrating the process)
print("\n🏷️ Label encoding...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Original labels: {np.unique(y)}")
print(f"Encoded labels: {np.unique(y_encoded)}")

# 4. Split Data
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# 5. Create and Train Pipeline
print("\n🌳 Training Decision Tree Classifier...")
# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# 6. Evaluate Model
print("\n📈 Model Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Iris Classification')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# ================================================================
# TASK 2: DEEP LEARNING WITH TENSORFLOW
# ================================================================

print("\n✅ TASK 2: Deep Learning with TensorFlow")
print("-" * 40)

# 1. Data Preparation
print("📊 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape for CNN (add channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"Reshaped training data: {x_train.shape}")
print(f"Pixel value range: [{x_train.min():.2f}, {x_train.max():.2f}]")

# 2. Model Architecture
print("\n🧠 Building CNN model...")
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

print("🏗️ Model Architecture:")
model.summary()

# 3. Compile and Train
print("\n⚙️ Compiling model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("🏋️ Training model...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 4. Evaluation
print("\n📊 Evaluating model...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# Check if we achieved >95% accuracy
if test_accuracy > 0.95:
    print("🎉 SUCCESS: Model achieved >95% accuracy!")
else:
    print(f"⚠️  Model achieved {test_accuracy*100:.2f}% accuracy (target: >95%)")

# Make predictions on test set
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Visualize some predictions
plt.figure(figsize=(12, 8))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'True: {y_test[i]}, Pred: {predicted_classes[i]}')
    plt.axis('off')
plt.suptitle('MNIST Predictions Sample')
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# ================================================================
# TASK 3: NLP WITH SPACY
# ================================================================

print("\n✅ TASK 3: NLP with spaCy")
print("-" * 40)

# Sample user reviews for demonstration
sample_reviews = [
    "I love this iPhone! The camera quality is amazing and the battery life is excellent.",
    "Samsung Galaxy phone is okay, but the screen is too bright. Not bad overall.",
    "Terrible experience with this MacBook. The keyboard is horrible and it's overpriced.",
    "Great laptop from Dell! Fast performance and good build quality. Highly recommend.",
    "The Nike shoes are comfortable but expensive. Mixed feelings about the purchase.",
    "Absolutely hate this Adidas product. Poor quality and terrible customer service.",
    "Sony headphones have incredible sound quality. Best purchase I've made this year!",
    "Microsoft Surface tablet is decent. Good for work but gaming performance is poor."
]

print("📝 Sample Reviews:")
for i, review in enumerate(sample_reviews, 1):
    print(f"{i}. {review}")

# Try to load spaCy model
try:
    print("\n🔤 Loading spaCy English model...")
    # Try to load the model
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully!")
except OSError:
    print("❌ spaCy English model not found. Please install it with:")
    print("python -m spacy download en_core_web_sm")
    print("\n🔄 Using alternative NLP approach...")
    nlp = None

# Define sentiment keywords (simple rule-based approach)
positive_words = {
    'love', 'amazing', 'excellent', 'great', 'good', 'best', 'incredible', 
    'fantastic', 'awesome', 'perfect', 'wonderful', 'outstanding', 'superb',
    'comfortable', 'recommend', 'fast', 'quality'
}

negative_words = {
    'hate', 'terrible', 'horrible', 'bad', 'poor', 'awful', 'worst', 
    'disappointing', 'useless', 'overpriced', 'expensive', 'slow',
    'broken', 'defective', 'annoying'
}

def analyze_sentiment_simple(text):
    """Simple rule-based sentiment analysis"""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count > negative_count:
        return 'positive', positive_count - negative_count
    elif negative_count > positive_count:
        return 'negative', negative_count - positive_count
    else:
        return 'neutral', 0

def extract_brands_simple(text):
    """Simple brand extraction using predefined list"""
    brands = ['iphone', 'samsung', 'galaxy', 'macbook', 'dell', 'nike', 
              'adidas', 'sony', 'microsoft', 'surface', 'apple']
    
    text_lower = text.lower()
    found_brands = []
    for brand in brands:
        if brand in text_lower:
            found_brands.append(brand.title())
    return found_brands

# Analyze reviews
print("\n🔍 Analyzing Reviews:")
print("=" * 60)

results = []

for i, review in enumerate(sample_reviews, 1):
    print(f"\nReview {i}: {review}")
    
    # Sentiment Analysis
    sentiment, score = analyze_sentiment_simple(review)
    print(f"💭 Sentiment: {sentiment.upper()} (score: {score})")
    
    # Brand/Entity Extraction
    if nlp is not None:
        # Use spaCy for named entity recognition
        doc = nlp(review)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        brands_spacy = [ent[0] for ent in entities if ent[1] in ['ORG', 'PRODUCT']]
        brands = list(set(brands_spacy + extract_brands_simple(review)))
    else:
        # Use simple brand extraction
        brands = extract_brands_simple(review)
    
    if brands:
        print(f"🏷️  Brands/Products: {', '.join(brands)}")
    else:
        print(f"🏷️  Brands/Products: None detected")
    
    results.append({
        'review': review,
        'sentiment': sentiment,
        'score': score,
        'brands': brands
    })
    print("-" * 60)

# Summary Statistics
print("\n📊 Analysis Summary:")
sentiments = [r['sentiment'] for r in results]
sentiment_counts = Counter(sentiments)

print(f"Total reviews analyzed: {len(results)}")
print(f"Sentiment distribution:")
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(results)) * 100
    print(f"  {sentiment.title()}: {count} ({percentage:.1f}%)")

# Brand mentions
all_brands = []
for r in results:
    all_brands.extend(r['brands'])
brand_counts = Counter(all_brands)

print(f"\nBrand mentions:")
for brand, count in brand_counts.most_common():
    print(f"  {brand}: {count}")

# Visualization
if sentiment_counts:
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    sentiments_list = list(sentiment_counts.keys())
    counts_list = list(sentiment_counts.values())
    colors = ['green' if s == 'positive' else 'red' if s == 'negative' else 'gray' 
              for s in sentiments_list]
    plt.bar(sentiments_list, counts_list, color=colors, alpha=0.7)
    plt.title('Sentiment Distribution')
    plt.ylabel('Number of Reviews')
    
    if brand_counts:
        plt.subplot(1, 2, 2)
        brands_list = list(brand_counts.keys())[:5]  # Top 5 brands
        brand_counts_list = [brand_counts[b] for b in brands_list]
        plt.bar(brands_list, brand_counts_list, color='skyblue', alpha=0.7)
        plt.title('Top Brand Mentions')
        plt.ylabel('Mentions')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

print("\n🎉 All tasks completed successfully!")
print("=" * 50)
print("Summary:")
print("✅ Task 1: Decision Tree achieved high accuracy on Iris dataset")
print(f"✅ Task 2: CNN achieved {test_accuracy*100:.1f}% accuracy on MNIST")
print("✅ Task 3: NLP analysis completed with sentiment and entity extraction")
