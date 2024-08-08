import os
import time
import pandas as pd
import google.generativeai as genai


def get_answers(text, question):
    prompt = f"""You are a researcher screening titles and abstracts of scientific papers for the systematic review '{review_title}'.
    Analyse the abstract below within the brackets and answer the question below. Taking a step-by-step approach towards reasoning and answering the question.
    Question: {question}
    Keep your answers as short as possible. The answer should only be in either a positive, neutral, or negative sentiment format.
    """
    response = model.generate_content([prompt, text], generation_config={"temperature": 0.2})
    return response.text.strip()


def main():
    # Get user inputs
    input_path = input("Enter the path of the input CSV file: ")
    api_key = input("Enter the Google Generative AI API key: ")
    output_csv_path = input("Enter the path where the output CSV file should be saved: ")

    # Configure genai with API key
    genai.configure(api_key=api_key)

    global model
    model = genai.GenerativeModel('gemini-pro')

    global review_title
    # Load data from CSV file
    data = pd.read_csv(input_path)
    review_title = data['Review Title'].iloc[0]

    # Get questions from the user
    questions = []
    print("Please enter 5 questions:")
    for i in range(5):
        question = input(f"Enter question {i + 1}: ")
        questions.append(question)

    list_abs = data['text'].tolist()

    # Initialize answer lists
    answers = {f'Gemini_question{i + 1}': [''] * len(list_abs) for i in range(5)}

    # Load existing partial data if available
    if os.path.exists(output_csv_path):
        partial_data = pd.read_csv(output_csv_path)
        for key in answers.keys():
            if key in partial_data.columns:
                answers[key] = partial_data[key].tolist()

    # Get answers for each abstract and each question
    for idx, text in enumerate(list_abs):
        for i, question in enumerate(questions):
            if answers[f'Gemini_question{i + 1}'][idx] == '':
                try:
                    answer = get_answers(text, question)
                    answers[f'Gemini_question{i + 1}'][idx] = answer
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
