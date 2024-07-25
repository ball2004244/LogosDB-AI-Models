'''
Generate synthetic data for training and testing
'''

from typing import List
from dotenv import load_dotenv
import os
import time
import pandas as pd
import google.generativeai as genai

# load input from input csv, generate synthetic keywords and save to output csv
# keywords col: "kw1, kw2, kw3, kw4, kw5"

# call Gemini API to generate keywords
#!pip install -q -U google-generativeai

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

def gen_keywords(text: str) -> List[str]:
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = '''
        Given this concat str from a question & answer, can you please extract top 5 keywords that best capture main idea. 
        You must do extractive summary.

        Dont say any thing else but the output. 
        Return format must be like this:
        [kw1, kw2, kw3, kw4, kw5]

        Here is the paragraph to be summarized:
    '''
    
    response = model.generate_content(f'{prompt}\n{text}')
    
    time.sleep(2)
    return eval(response.text)

def gen_data(input_csv, output_csv):
    input_data = pd.read_csv(input_csv)
    output_data = input_data.copy()
    output_data['Keywords'] = output_data['RawText'].apply(lambda x: gen_keywords(x))
    output_data.to_csv(output_csv, index=False)
    
    
def main():
    input_csv = 'single_qna.csv'
    output_csv = 'synthesis_inp.csv'
    gen_data(input_csv, output_csv)
    
if __name__ == "__main__":
    main()