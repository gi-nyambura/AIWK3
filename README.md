# AIWK3
# 🚀 Complete Machine Learning Pipeline

This project demonstrates a comprehensive machine learning workflow covering **classical machine learning**, **deep learning**, and **natural language processing (NLP)** using popular Python frameworks.

---

## 📌 **Tasks Overview**

### ✅ **Task 1: Classical ML with Scikit-learn**
- **Dataset:** Iris Species Dataset
- **Goal:** Build a robust Decision Tree Classifier to predict iris species.
- **Steps:**
  - Load and explore the Iris dataset
  - Handle missing values (demonstrated)
  - Encode labels (demonstrated)
  - Preprocess features (imputation, scaling)
  - Train a Decision Tree within a `Pipeline`
  - Evaluate using accuracy, precision, recall, classification report, and confusion matrix

### ✅ **Task 2: Deep Learning with TensorFlow**
- **Dataset:** MNIST Handwritten Digits
- **Goal:** Train a Convolutional Neural Network (CNN) to classify handwritten digits with >95% test accuracy.
- **Steps:**
  - Load and preprocess MNIST data
  - Build a deep CNN with multiple convolutional layers
  - Compile and train the model
  - Evaluate test accuracy and visualize predictions
  - Plot training history (accuracy & loss)

### ✅ **Task 3: NLP with spaCy**
- **Dataset:** Sample Amazon-style product reviews
- **Goal:** Perform Named Entity Recognition (NER) and simple sentiment analysis.
- **Steps:**
  - Load or fallback to rule-based NLP if `spaCy` model is missing
  - Extract brands and products mentioned in reviews
  - Apply simple rule-based sentiment analysis (positive, negative, neutral)
  - Visualize sentiment distribution and top brand mentions

---

## ⚙️ **Requirements**

Install the following Python packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow spacy
python -m spacy download en_core_web_sm

### 📂 Project Structure
- pipeline.py — Main Python script containing the complete workflow

- README.md — This file

- requirements.txt — Recommended dependencies list (optional)

- Outputs:

      - Confusion matrix plot (Iris)
      
      - CNN training history plots (MNIST)
      
      - Sentiment and brand distribution plots (NLP)


🏃 How to Run
1. Clone this repository or download the script.

2. Install dependencies.

3. Run the script:

python pipeline.py

4. Review the printed summaries and generated visualizations.

🧩 Key Frameworks

| Library        | Purpose                           |
| -------------- | --------------------------------- |
| `scikit-learn` | Classical machine learning models |
| `tensorflow`   | Deep learning with CNN            |
| `spaCy`        | Advanced NLP tasks                |
| `matplotlib`   | Data visualization                |
| `seaborn`      | Enhanced plots                    |

✅ Results Summary
- Task 1: Decision Tree Classifier achieves high accuracy on the Iris dataset.

- Task 2: CNN model achieves target accuracy (>95%) on MNIST.

- Task 3: NLP pipeline extracts brands and infers sentiment from user reviews.

👩‍💻 Author
Patricia Nyambura

📜 License
This project is intended for educational purposes. Feel free to adapt and reuse!


