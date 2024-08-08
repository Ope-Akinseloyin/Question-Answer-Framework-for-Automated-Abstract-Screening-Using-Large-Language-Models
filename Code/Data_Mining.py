import pandas as pd
from metapub import PubMedFetcher

def main():
    # Get user inputs
    input_path = input("Enter the path of the input CSV file: ")
    review_title = input("Enter the review title: ")
    review_criteria = input("Enter the review criteria: ")
    output_path = input("Enter the path where the output CSV file should be saved: ")

    # Load data from CSV file
    data = pd.read_csv(input_path, index_col=0)

    # Add review title and criteria to the data
    data['Review Title'] = review_title
    data['Review Criteria'] = review_criteria

    # Get the list of PMIDs
    pmids = data['PMID'].to_list()

    # Initialize dictionaries to hold titles and abstracts
    titles = {}
    abstracts = {}

    # Fetch titles and abstracts from PubMed
    fetcher = PubMedFetcher()
    for pmid in pmids:
        try:
            article = fetcher.article_by_pmid(pmid)
            titles[pmid] = article.title
            abstracts[pmid] = article.abstract
        except Exception as e:
            print(f"An error occurred while fetching data for PMID {pmid}: {e}")

    # Create DataFrames from the fetched data
    title_df = pd.DataFrame(list(titles.items()), columns=['PMID', 'Title'])
    abstract_df = pd.DataFrame(list(abstracts.items()), columns=['PMID', 'Abstract'])

    # Merge the original data with the fetched titles and abstracts
    enriched_data = pd.merge(data, title_df, on='PMID', how='inner')
    enriched_data = pd.merge(enriched_data, abstract_df, on='PMID', how='inner')

    # Save the enriched data to the specified output CSV file
    enriched_data.to_csv(output_path)
    print(f"Data successfully saved to {output_path}")

if __name__ == "__main__":
    main()