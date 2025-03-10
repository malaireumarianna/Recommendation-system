# Recommendation-system

Structure:
```
├── LLama-book-descr_generat.ipynb              
│   
├── NN_recommendation_system.ipynb          
│   
├── recommendation_system-KNN.ipynb
│              
├── recommendation_system_ALS.ipynb
```

The content of notebooks can be viewed via nbviewer:

**https://nbviewer.org/github/malaireumarianna/Recommendation-system/tree/main/**



## **LLama-book-descr_generat.ipynb description**

The notebook's code generates book descriptions using a large language model (LLM), specifically a model from the Meta-LLaMA family `Llama-3.2-1B`. It utilizes the Hugging Face transformers library for model loading and inference and processes book metadata (title, author) to create short descriptions.


## **recommendation_system_ALS.ipynb description**

Implements a Collaborative Filtering-based book recommendation system using Alternating Least Squares (ALS), a matrix factorization technique provided by the `implicit` library. The system processes a dataset of book ratings, trains an ALS model, and recommends books based on past ratings to users.


## **recommendation_system-KNN.ipynb description**

This notebook's approach processes book descriptions generated in different files from the book dataset by cleaning the text, removing stopwords, and lemmatizing words to create a new preprocessed version of the descriptions. The script uses `NLTK (Natural Language Toolkit)` for text processing.
The code implements a content-based book recommendation system using TF-IDF features and nearest neighbor search. Using cosine similarity, the goal is to find books with descriptions similar to those of a given book.


## **recommendation_system-KNN.ipynb description**

This notebook trains a deep learning-based book recommendation system using PyTorch. It incorporates both collaborative filtering (user interactions) and content-based filtering (book descriptions), utilizing a custom PyTorch Dataset (BookDataset) to structure the training data. The goal is to predict book ratings and recommend books based on learned patterns.
