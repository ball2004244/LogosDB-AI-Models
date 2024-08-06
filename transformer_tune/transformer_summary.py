from transformers import T5Tokenizer, T5ForConditionalGeneration
import time

# Load the fine-tuned model and tokenizer
model_name = "./finetuned_model"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_texts(texts):
    # Prepare the inputs
    inputs = [f"summarize: {text}" for text in texts]
    input_ids = tokenizer(inputs, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    # Generate the summaries
    summary_ids = model.generate(input_ids["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the outputs
    summaries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
    return summaries

if __name__ == '__main__':
    start_time = time.perf_counter()
    input_file = 'summary_input.txt'
    output_file = 'summary_output.txt'

    print('Reading user inputs from file...')
    user_inputs = []
    with open(input_file, 'r') as file:
        user_inputs = file.readlines()


    print('Summarizing user inputs...')
    # Get the summaries
    summaries = summarize_texts(user_inputs)
    for i, summary in enumerate(summaries):
        print(f"Summary {i+1}: {summary}")
        
    print('Writing summaries to file...')
    with open(output_file, 'w') as file:
        for summary in summaries:
            file.write(summary + '\n')
            
    print('Summarization complete!')
    print(f'Time taken: {time.perf_counter() - start_time:.2f} seconds')
    
    # Analysis
    '''
    1 row: 1.02s
    2 row: 2.72s
    5 row: 4.83s
    10 row: 11.12s
    50 row: 48.09s
    '''