# Sentiment Analysis using NLP

## Table of Contents
1. [Abstract](#abstract)
2. [Data Acquisition](#data-acquisition)
3. [Data Exploration](#data-exploration)
4. [Data Preparation](#data-preparation)
5. [Model Building](#model-building)
    - [Machine Learning Model](#machine-learning-model)
    - [Deep Learning Model](#deep-learning-model)
6. [Model Evaluation](#model-evaluation)
7. [Model Saving](#model-saving)
8. [Model Testing](#model-testing)
9. [Contributing](#contributing)
10. [Author](#author)


## Abstract
Developed a sentiment analysis pipeline to classify text into positive, negative, or neutral categories. utilizing NLTK for text preprocessing and TF-IDF for feature extraction. Implemented two models: a machine learning model using Naive Bayes, targeting approximately 93% validation accuracy and 91% testing accuracy, and a deep learning model using LSTM, with both validation and testing accuracies around 92%

## Data Acquisition
Link: [Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset)

The Social Media Sentiments Analysis Dataset captures a vibrant tapestry of emotions, trends, and interactions across various social media platforms. This dataset provides a snapshot of user-generated content, encompassing text, timestamps, hashtags, countries, likes, and retweets. Each entry unveils unique stories—moments of surprise, excitement, admiration, thrill, contentment, and more—shared by individuals worldwide.

### Features
- **Text**: User-generated content showcasing sentiments.
- **Sentiment**: Categorized emotions.
- **Timestamp**: Date and time information.
- **User**: Unique identifiers of users contributing.
- **Platform**: Social media platform where the content originated.
- **Hashtags**: Identifies trending topics and themes.
- **Likes**: Quantifies user engagement (likes).
- **Retweets**: Reflects content popularity (retweets).
- **Country**: Geographical origin of each post.
- **Year**: Year of the post.
- **Month**: Month of the post.
- **Day**: Day of the post.
- **Hour**: Hour of the post.

The data is split into training (70%), validation (15%), and testing (15%) sets.

## Data Exploration
- Check for missing values.
- Remove duplicate entries.
- Clean up sentiment labels by removing leading and trailing whitespace.
- Map sentiments to positive, negative, and neutral categories.
- Address class imbalance using Random OverSampler.

## Data Preparation
Perform text preprocessing: tokenization, removing stopwords, stemming/lemmatization.

## Model Building

### Machine Learning Model
1. Build a pipeline with TF-IDF vectorization and Naive Bayes classifier.
2. Perform hyperparameter tuning using RandomizedSearchCV.
3. Train the model with the best parameters.

### Deep Learning Model
1. Implement a deep learning model using LSTM.
2. Tokenize texts and convert them to sequences.
3. Pad sequences to ensure uniform length.
4. Build and train the LSTM model.

## Model Evaluation
- Evaluate the machine learning model using validation data.
- Generate classification reports and confusion matrices.

## Model Saving
- Save the trained model using joblib.

## Model Testing
- Test the machine learning model on test data.
- Make predictions on new examples to verify model performance.

## Contributing

Contributions are welcome! If you have suggestions, improvements, or additional content to contribute, feel free to open issues, submit pull requests, or provide feedback. 

[![GitHub watchers](https://img.shields.io/github/watchers/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Watch)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/watchers/?WT.mc_id=academic-105485-koreyst)
[![GitHub forks](https://img.shields.io/github/forks/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Fork)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/network/?WT.mc_id=academic-105485-koreyst)
[![GitHub stars](https://img.shields.io/github/stars/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Star)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/stargazers/?WT.mc_id=academic-105485-koreyst)

## Author

This repository is maintained by Elsayed Elmandoh, an AI Engineer. You can connect with Elsayed on [LinkedIn and Twitter/X](https://linktr.ee/elsayedelmandoh) for updates and discussions related to Machine learning, deep learning and NLP.

Happy coding!

