# Unveiling Online Deception : Fake-review-detection using Deep Learning Techniques

This is a thesis project done as part of my Masters in Data Science Program in Liverpool John Moore University, UK.

**The research topic** : Fake Review Detection using Deep Learning Techniques.

**Aim and Objectives**

The main aim of this research is to study various machine learning and deep learning techniques used for fake review detection and come up with novel approach that can be applied using supervised or semi supervised methods for unlabeled data.

• To analyze various Embedding Techniques Machine Learning and Deep Learning techniques for fake review detection.
• To propose a suitable feature extraction method and model architecture for fake review detection.
• To assess the effectiveness of different Embedding Techniques such as BERT, RoBERTa, DistilBERT, XLNET and machine learning algorithms, including Support Vector Machine, Naïve-Bayes, and convolutional neural networks (CNNs), recurrent neural networks (RNNs), LSTM, BERT (transformer),  Adversarial Learning for automatic feature learning and complex language processing.
• To evaluate the overall performance of the proposed framework using appropriate metrics and validation techniques to assess its efficiency and accuracy in fake review detection.

**Data set** : Yelp data set is used for this thesis.

**The research methodology framework**

![image](https://github.com/srinay2007/Fake-review-detection/assets/98680554/3940c88d-df28-4324-b0a0-37e2beaaaf84)

Figure : Research Methodology Framework

**Embedding Techniques:** 
Traditional feature engineering may involve creating numerical features based on domain knowledge or extracting statistical properties from raw data. However, in NLP tasks, raw text data cannot be directly fed into machine learning models. Embedding techniques like **Word2Vec, GloVe, BERT** and others transform words or text sequences into dense numerical vectors, capturing semantic and contextual information.

TF-IDF: (Term Frequency Inverse document frequency)
Focuses on identifying important words in a document. It considers how often a word appears in the document (frequency) and how rare it is overall (inverse document frequency). This helps prioritize keywords that are specific and essential to the document's content.

GloVe: (Global Vectors for Word Representation) 
Captures semantic relationships between words. It analyzes a massive amount of text data to see how often words co-occur, essentially learning that words appearing together frequently share similar meanings. This allows tasks like finding synonyms or analogies based on word vector similarities.

BERT (Bidirectional Encoder Representations from Transformers) : 
•	A powerful technique for understanding words based on their context in a sentence. It's pre-trained on a huge dataset of text and code, allowing it to learn contextual word representations. 
•	BERT can be used as an embedding technique to generate contextualized word embeddings for text data. Unlike traditional word embeddings like Word2Vec or GloVe, which produce fixed representations for each word regardless of context, BERT captures contextual information by considering the entire sentence bidirectionally.

![image](https://github.com/srinay2007/Fake-review-detection/assets/98680554/c4d9ba5c-3415-4635-9cec-45581968d8b4)
 
Figure: BERT Illustration

Source: https://www.researchgate.net/publication/359301499_Deep_learning-based_methods_for_natural_hazard_named_entity_recognition/figures?lo=1

**RoBERTta**: Builds on BERT's success, aiming for better efficiency and performance. It utilizes a more sophisticated masking strategy during training and removes unnecessary steps, making the training process faster and potentially improving performance on certain NLP tasks while retaining BERT's core strengths.

**DistilBERT**: Creates a compact and efficient version of BERT through a technique called knowledge distillation. It learns from a larger pre-trained model (like BERT) but with a significantly smaller size, allowing for faster processing and lower resource requirements, while maintaining good performance on NLP tasks

**XLNet**: Addresses limitations in BERT's pre-training process by considering all possible permutations of ordering the input words. This can potentially capture more nuanced relationships between words compared to BERT, but it comes with a more complex architecture and even higher computational demands.
 
![image](https://github.com/srinay2007/Fake-review-detection/assets/98680554/b472d56e-72e9-4770-b609-c7bc36847ff4)

Figure: Comparison of Embedding techniques.

**Implementation**

This thesis has implemented  the Machine Learning algorithms, namely **Logistic Regression, Support Vector Machine, Random Forest, XGBoost** and Deep Learning Models such as **CNN (Convolution Neural Network), CNN-LSTM, RNN -LSTM, BERT (Bidirectional Encoder Representations from Transformers)** applied for the fake review detection. During this process various embedding techniques such as **TFIDF, Glove, BERT, ROBERTA, DISTEILBERT, XLNET** embeddings were explored. Balancing techniques such as ADASYN, SMOTE (Oversampling minority classes) were tested. Standard Scaler were applied for numerical columns. It is noted that Deep Learning Models MLP Classifier with BERT, ROBERTA, DISTEILBERT, XLNET Embeddings outperformed other Embedding techniques TFIDF and Glove on Machine Learning Classifiers.

The codes are organized into 4 folder 
1. Tfidf - 5 ML Models tested with Tfidf embedding technique
2. Glove - 5 ML Models tested with Glove embedding technique
3. BERT - SVM Classifier tested with BERT and MLP Classifier tested with BERT, ROBERTA, DISTEILBERT, XLNET embeddnig techniques. 
4. DL Classifiers - CNN, CNN-LSTM and RNN-LSTM classifiers are tested.




