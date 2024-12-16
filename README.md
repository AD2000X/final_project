# Sensational Language Identification Using Machine Learning and NLP

## 1. Abstract
This research pioneers the application of machine learning and natural language processing (NLP) to identify sensational language in news headlines using a self-created dataset, **MIRUKU**. It highlights sensational language discussions and identifies key features through **SHAP analysis**.

---

## 2. Definition of Sensational
**Sensational Language**: Expressions that quickly evoke emotional arousal and interest through vivid, exaggerated, dramatic wording or shocking content, appealing to curiosity, emotions, or biases.

---

## 3. Background of the Study
### News
News headlines act as **relevance optimizers** to capture readers’ attention and influence decisions to engage with content. Modern news strategies focus on increasing **Click-Through Rates (CTR)**, with sensational headlines being a key tactic.

### Language
Language plays a crucial role in communication, promoting **emotional engagement**. Sensationalism triggers emotional and psychological responses, helping to retain attention.

### Sensational
Sensational language emphasizes **emotional and physiological responses**, aligning with human evolutionary traits of attention to threats or survival-related information.

### Evolution
Human evolution has shaped an inherent tendency to respond to **sensational stimuli** for survival and reproduction. This explains why sensational news effectively draws attention.

---

## 4. Problem Statement
Most research on sensationalism focuses on **corpus analysis** rather than **NLP methods**.  
NLP combined with machine learning offers an efficient way to identify sensational language and analyze its contributing elements.

---

## 5. Purpose of the Study
### Linguistics Transition
News headlines often blend multiple categories, requiring a **stylistic and analytical approach** to identify sensationalism. The study aims to analyze linguistic features in sensational news.

---

## 6. Research Questions or Hypotheses
1. Can a **single linguistic feature** effectively and stably identify sensational language?  
2. Which **algorithm** performs best in detecting sensational language?  
3. What **features** are most effective in identifying sensational language?

---

## 7. Scope and Delimitations
### Included Features
The study explores multiple linguistic features, including:
- Stop word ratios
- TF-IDF (with and without stop words)
- Syntactic n-grams
- Subjectivity and sentiment analysis
- Readability scores
- Elongated words
- Punctuation marks (e.g., exclamation marks, ellipses)

### Excluded Features
- Named Entity Recognition (NER)
- Part-of-Speech (POS) tags
- Clustering algorithms

### Limitations
- **Reproducibility**: Choices of methods, such as **Cuckoo Search** for feature extraction and **Random Search** for training, may affect reproducibility.
- **Threshold Adjustment**: Adjusted thresholds were not tested in follow-up experiments.


