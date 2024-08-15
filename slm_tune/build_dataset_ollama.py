from typing import List
import requests
import time
import pandas as pd
'''
This file is to build dataset for finetuning Qwen2 model. It uses synthetic data from LLama 3.1 to build dataset.
'''
OLLAMA_URL = "http://localhost:11434/api/generate"

def build_input(raw_file: str, temp_file: str = 'temp.csv') -> None:
    raw_data = pd.read_csv(raw_file)

    # header: Question, Answer
    # Concat each question and answer into 1 string, denoted by "Question: <question>\nAnswer: <answer>\n"

    data = []
    for i in range(len(raw_data)):
        question = raw_data['Question'][i]
        answer = raw_data['Answer'][i]
        data.append(f"Question: {question}\nAnswer: {answer}\n")
        
    # Write the data to a csv file with only the 'Input' column
    df = pd.DataFrame(data, columns=['Input'])
    df.to_csv(temp_file, index=False)

def call_llama(prompt: str) -> List[str]:
    model = "llama3.1:8b"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    # Call llama API to generate summaries
    res = response.json()['response']
        
    # Convert response back to list, each item starts with '-'
    summarized_list = [item.strip() for item in res.split('\n-') if item.strip()]
    return summarized_list

import pandas as pd

def build_output(out_file: str, temp_file: str = 'temp.csv', CHUNK_SIZE: int = 100, max_chunk_count: int = -1) -> None:
    # Initialize an empty list to store the outputs
    outputs = []
    inputs = []
    RETRIES = 5
    count = 0

    try:
        for chunk in pd.read_csv(temp_file, chunksize=CHUNK_SIZE):
            if max_chunk_count != -1 and count >= max_chunk_count:
                print(f"Reached max chunk count of {max_chunk_count}. Stopping.")
                break

            # Concatenate all rows in the current chunk into a single string with a dash before each point
            input_text = ' '.join([f"- {item}" for item in chunk['Input'].tolist()])
        
            # Create the prompt
            prompt = f'''
            Summarize below {CHUNK_SIZE} bullets paragraph into {CHUNK_SIZE} single-sentence summaries as 1-on-1 mapping.
            Answer nothing but the summaries. Don't include this phrase too: "Here are the summarized answers:".
        
            You must strictly follow the format below:
            ```
            - <Summary 1>
            - <Summary 2>
            - <Summary 3>
            ...
            ```
        
            Only respond with strictly {CHUNK_SIZE} summaries.
            Inputs for summarize:
            {input_text}
            '''
        
            # Retry mechanism
            success = False
            for attempt in range(RETRIES):
                try:
                    print(f"PROCESS CHUNK {count + 1}, ATTEMPT {attempt + 1}, CHUNK_SIZE: {CHUNK_SIZE}, MAX_CHUNK: {max_chunk_count}")
                    # Call the LLM with the concatenated string
                    output = call_llama(prompt)
                    
                    # compare the length of output and chunk
                    if len(output) != CHUNK_SIZE:
                        raise Exception(f"Output length mismatch: expected {CHUNK_SIZE}, got {len(output)}")

                    success = True
                    break  # Exit the loop if successful
                except Exception as e:
                    print(f"FAILED CHUNK {count + 1}, ATTEMPT {attempt + 1}")
            
            if success:
                # Append each input and its corresponding output to the lists
                inputs.extend(chunk['Input'].tolist())
                outputs.extend(output)
                print(f"Finished processing chunk {count + 1}")
                count += 1
            else:
                print(f"Skipping chunk {count + 1} after {RETRIES} failed attempts")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Saving checkpoint...")
        output_df = pd.DataFrame({'Input': inputs, 'Output': outputs})
        output_df.to_csv(out_file, index=False)
        print("Checkpoint saved. Exiting program.")
        raise

    # Save final inputs and outputs to a new CSV file with two columns: Input and Output
    output_df = pd.DataFrame({'Input': inputs, 'Output': outputs})
    output_df.to_csv(out_file, index=False)

if __name__ == "__main__":
    # 1.2M row / 10 = 120k chunks
    start = time.perf_counter()
    print("Building dataset...")
    in_file = 'raw.csv'
    temp_file = 'temp.csv'
    out_file = 'train.csv'
    CHUNK_SIZE = 10
    max_chunk_count = 3000 # stop when reach 3000 chunks ~ 30k rows, 18987.21s ~ 5.3 hours
    
    print(f"Build input file: {in_file}")
    build_input(in_file, temp_file)
    print("Input built successfully")
    
    print(f"Build output file: {out_file}")
    build_output(out_file, temp_file, CHUNK_SIZE, max_chunk_count)
    print("Output built successfully")

    print("Dataset built successfully!!!")
    print(f"Time taken: {time.perf_counter() - start:.2f} seconds")