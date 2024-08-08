import numpy as np
import pandas as pd
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

def embed_text(model_name, task_type, text, title="", output_dimensionality=None):
    """Generates a text embedding with a Large Language Model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    text_embedding_input = TextEmbeddingInput(task_type=task_type, title=title, text=text)
    kwargs = dict(output_dimensionality=output_dimensionality) if output_dimensionality else {}
    embeddings = model.get_embeddings([text_embedding_input], **kwargs)
    return embeddings[0].values

def calculate_cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    # Get user inputs
    input_path = input("Enter the path of the input CSV file: ")
    api_key = input("Enter the GPT API key: ")
    output_path = input("Enter the path where the output CSV file should be saved: ")

    # Load data from CSV file
    data = pd.read_csv(input_path)
    criteria = data['Review Criteria'].iloc[0]

    # Constants for the embedding model
    MODEL = "text-embedding-004"
    TASK = "RETRIEVAL_DOCUMENT"
    OUTPUT_DIMENSIONALITY = 768

    # Get embeddings for the text in the CSV
    text_list = data['text'].tolist()
    embeddings = [embed_text(MODEL, TASK, text) for text in text_list]

    # Add embeddings to the DataFrame
    data['Gemini_embedding'] = embeddings

    # Get embedding for the criteria
    criteria_embed = embed_text(MODEL, 'RETRIEVAL_QUERY', criteria)

    # Calculate and add cosine similarity scores
    data['Gemini_criteria_score'] = data['Gemini_embedding'].apply(lambda x: calculate_cosine_similarity(x, criteria_embed))

    # Save the enriched data to the specified output CSV file
    data.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")

if __name__ == "__main__":
    main()
