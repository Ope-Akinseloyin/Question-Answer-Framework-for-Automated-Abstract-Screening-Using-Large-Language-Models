import numpy as np
import pandas as pd
from openai import OpenAI
import os
import time

def main():
    # Get user inputs
    input_path = input("Enter the path of the input CSV file: ")
    api_key = input("Enter the GPT API key: ")
    output_csv_path = input("Enter the path where the output CSV file should be saved: ")
    output_txt_path = input("Enter the path where the output questions text file should be saved: ")

    questions_input = input("Do you want to input the questions directly? (yes/no): ").strip().lower()

    # Load data from CSV file
    data = pd.read_csv(input_path)
    criteria = data['Review Criteria'].iloc[0]
    review_title = data['Review Title'].iloc[0]

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    if questions_input == 'yes':
        questions = []
        print("Please enter 5 questions:")
        for i in range(5):
            question = input(f"Enter question {i+1}: ")
            questions.append(question)
    else:
        # Generate questions based on the review criteria
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a researcher screening titles and abstracts of scientific papers"},
                {"role": "user", "content": f"Using the inclusion criteria in the brackets generate 5 unique yes or no questions which encompass the entire inclusion criteria without any duplicate or unnecessary questions to ascertain if papers meet the inclusion criteria ({criteria})"}
            ],
            temperature=0.2,
            max_tokens=1024
        )

        questions = response.choices[0].message.content.strip().split('\n')
    
    with open(output_txt_path, 'w') as f:
        for question in questions:
            f.write(question + '\n')

    # Get embeddings for the questions
    question_embeddings = [client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding for question in questions]

    # Convert stored embeddings from string to list
    data['GPT_embedding'] = data['GPT_embedding'].apply(eval)

    # Calculate cosine similarity
    def calculate_cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for i, question_embed in enumerate(question_embeddings):
        data[f'GPT_query{i+1}'] = data['GPT_embedding'].apply(lambda x: calculate_cosine_similarity(x, question_embed))

    # Function to ask questions and get answers
    def get_answers(text, question):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": f"You are a researcher screening titles and abstracts of scientific papers for the systematic review '{review_title}'."},
                {"role": "user", "content": f"Analyse the abstract below within the brackets and answer the question below. Taking a step-by-step approach towards reasoning and answering the question.\nAbstract:({text})\nQuestion: {question}\nKeep your answers as short as possible. The answer should be in either a positive, neutral, or negative sentiment format."}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()

    list_abs = data['text'].tolist()

    # Initialize answer lists
    answers = {f'GPT_question{i+1}': [''] * len(list_abs) for i in range(5)}

    # Load existing partial data if available
    if os.path.exists(output_csv_path):
        partial_data = pd.read_csv(output_csv_path)
        for key in answers.keys():
            if key in partial_data.columns:
                answers[key] = partial_data[key].tolist()

    # Get answers for each abstract and each question
    for idx, text in enumerate(list_abs):
        for i, question in enumerate(questions):
            if answers[f'GPT_question{i+1}'][idx] == '':
                try:
                    answer = get_answers(text, question)
                    answers[f'GPT_question{i+1}'][idx] = answer
                except Exception as e:
                    print(f"Error processing abstract: {text}\nQuestion: {question}\nError: {e}")
                    # Save data so far to avoid losing progress
                    partial_data = data.copy()
                    for key, value in answers.items():
                        partial_data[key] = value
                    partial_data.to_csv(output_csv_path, index=False)
                    print(f"Partial data saved to {output_csv_path}. Retrying in 5 seconds...")
                    time.sleep(5)

    # Add answers to the dataframe
    for key, value in answers.items():
        data[key] = value

    # Save the enriched data to the specified output CSV file
    data.to_csv(output_csv_path, index=False)
    print(f"Data successfully saved to {output_csv_path}")

if __name__ == "__main__":
    main()
