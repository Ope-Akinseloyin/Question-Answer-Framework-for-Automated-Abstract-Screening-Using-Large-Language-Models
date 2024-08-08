import numpy as np
import pandas as pd
from openai import OpenAI

def main():
    # Get user inputs
    input_path = input("Enter the path of the input CSV file: ")
    api_key = input("Enter the GPT API key: ")
    output_path = input("Enter the path where the output CSV file should be saved: ")

    # Load data from CSV file
    data = pd.read_csv(input_path, index_col=0)
    data.dropna(inplace=True)

    # Concatenate Abstract and Title into a single text field
    data['text'] = data['Abstract'] + ' ' + data['Title'] + '.'
    criteria = data['Review Criteria'].iloc[0]
    data.reset_index(inplace=True)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Function to get embeddings
    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input=text, model=model).data[0].embedding

    # Apply the get_embedding function to the text column
    data['GPT_embedding'] = data['text'].apply(lambda x: get_embedding(x))

    # Get the embedding for the criteria
    criteria_embed = get_embedding(criteria)

    # Function to calculate cosine similarity
    def calculate_cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Function to score the embeddings against the criteria
    def label_score(review_embedding):
        return calculate_cosine_similarity(review_embedding, criteria_embed)

    # Apply the label_score function to the embeddings
    data['GPT_criteria_score'] = data['GPT_embedding'].apply(label_score)

    # Save the data to the specified output CSV file
    data.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")

if __name__ == "__main__":
    main()