import requests

def call_rag(query: str, k: int=5) -> str:
    '''
    Helper function to call LogosDB as RAG model.
    '''
    # Define the endpoint URL
    url = "http://localhost:8000/smart_query"

    # Define the JSON payload
    payload = {
        "query": query,
        "k": k
    }

    # Make the HTTP GET request
    response = requests.get(url, headers={"Content-Type": "application/json"}, json=payload)

    # convert the response to JSON
    return response.json()['results']

if __name__ == '__main__':
    query = "What is the capital of France?"
    res = call_rag(query)
    print(res)
    # Output: {'query': 'What is the capital of France?', 'k': 5, 'answers': ['Paris', 'Marseille', 'Lyon', 'Toulouse', 'Nice']}