import pandas as pd
from transformers import pipeline

def soft_sentiment_score(text, classifier, candidate_labels):
    classification = classifier(text, candidate_labels)
    if classification['labels'][0] == 'Positive':
        return classification['scores'][0]
    else:
        return classification['scores'][1]

def hard_sentiment_score(text, classifier, candidate_labels):
    classification = classifier(text, candidate_labels)
    if classification['labels'][0] == 'Positive':
        return 1
    elif classification['labels'][0] == 'Neutral':
        return 0.5
    else:
        return 0

def rank_and_sort_csv(input_path, output_csv_path, ranking_mode, sentiment_mode, qa_model, embedding_model):
    # Load data from CSV file
    data = pd.read_csv(input_path)
    
    # Initialize the classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    
    # Define candidate labels based on the sentiment mode
    if sentiment_mode == 'soft':
        candidate_labels = ['Positive', 'Negative']
        sentiment_score = soft_sentiment_score
    elif sentiment_mode == 'hard':
        candidate_labels = ['Positive', 'Neutral', 'Negative']
        sentiment_score = hard_sentiment_score
    else:
        raise ValueError("Invalid sentiment mode. Choose from 'soft' or 'hard'.")
    
    # Calculate sentiment scores
    data['sentiment1'] = data[f'{qa_model}_question1'].apply(lambda x: sentiment_score(x, classifier, candidate_labels))
    data['sentiment2'] = data[f'{qa_model}_question2'].apply(lambda x: sentiment_score(x, classifier, candidate_labels))
    data['sentiment3'] = data[f'{qa_model}_question3'].apply(lambda x: sentiment_score(x, classifier, candidate_labels))
    data['sentiment4'] = data[f'{qa_model}_question4'].apply(lambda x: sentiment_score(x, classifier, candidate_labels))
    data['sentiment5'] = data[f'{qa_model}_question5'].apply(lambda x: sentiment_score(x, classifier, candidate_labels))
    
    # Calculate the score based on the ranking mode
    if ranking_mode == 'QA':
        data['score'] = (data['sentiment1'] + data['sentiment2'] + data['sentiment3'] + data['sentiment4'] + data['sentiment5']) / 5
    elif ranking_mode == 'criteria':
        data['score'] = (((data['sentiment1'] + data['sentiment2'] + data['sentiment3'] + data['sentiment4'] + data['sentiment5']) / 5) + data[f'{embedding_model}_criteria_score']) / 2
    elif ranking_mode == 'question':
        data['score'] = (((data['sentiment1'] + data[f'{embedding_model}_query1']) / 2 +
                          (data['sentiment2'] + data[f'{embedding_model}_query2']) / 2 +
                          (data['sentiment3'] + data[f'{embedding_model}_query3']) / 2 +
                          (data['sentiment4'] + data[f'{embedding_model}_query4']) / 2 +
                          (data['sentiment5'] + data[f'{embedding_model}_query5']) / 2) / 5)
    elif ranking_mode == 'composite':
        data['score'] = ((((data['sentiment1'] + data[f'{embedding_model}_query1']) / 2 +
                           (data['sentiment2'] + data[f'{embedding_model}_query2']) / 2 +
                           (data['sentiment3'] + data[f'{embedding_model}_query3']) / 2 +
                           (data['sentiment4'] + data[f'{embedding_model}_query4']) / 2 +
                           (data['sentiment5'] + data[f'{embedding_model}_query5']) / 2) / 5) + data[f'{embedding_model}_criteria_score']) / 2
    elif ranking_mode == 'Question Cosine':
        data['score'] = (data[f'{embedding_model}_query1'] + data[f'{embedding_model}_query2'] + data[f'{embedding_model}_query3'] + data[f'{embedding_model}_query4'] + data[f'{embedding_model}_query5']) / 5
    elif ranking_mode == 'Composite Cosine':
        data['score'] = (((data[f'{embedding_model}_query1'] + data[f'{embedding_model}_query2'] + data[f'{embedding_model}_query3'] + data[f'{embedding_model}_query4'] + data[f'{embedding_model}_query5']) / 5) + data[f'{embedding_model}_criteria_score']) / 2
    elif ranking_mode == 'Criteria Cosine':
        data['score'] = data[f'{embedding_model}_criteria_score']
    else:
        raise ValueError("Invalid ranking mode. Choose from 'QA', 'criteria', 'question', 'composite', 'Question Cosine', 'Composite Cosine', or 'Criteria Cosine'.")
    
    # Sort the data by score in descending order
    data_sorted = data.sort_values(by='score', ascending=False)
    
    # Save the sorted data to the specified output path with mode names in the filename
    output_filename = f"{output_csv_path.split('.')[0]}_{ranking_mode}_{sentiment_mode}_{qa_model}_{embedding_model}.csv"
    data_sorted.to_csv(output_filename, index=False)
    print(f"Data successfully ranked and saved to {output_filename}")

if __name__ == "__main__":
    input_path = input("Enter the path of the input CSV file: ")
    output_csv_path = input("Enter the path where the output CSV file should be saved: ")
    ranking_mode = input("Enter the ranking mode (QA, criteria, question, composite, Question Cosine, Composite Cosine, Criteria Cosine): ")
    sentiment_mode = input("Enter the sentiment scoring mode (soft, hard): ")
    qa_model = input("Enter the QA model name (Gemini, GPT, Llama, Claude): ")
    embedding_model = input("Enter the Embedding model name (Gemini, GPT, Llama, Claude): ")
    
    rank_and_sort_csv(input_path, output_csv_path, ranking_mode, sentiment_mode, qa_model, embedding_model)
