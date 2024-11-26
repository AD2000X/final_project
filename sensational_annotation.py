!pip install openai --quiet

from google.colab import drive
import glob
import os
import json
import pandas as pd
import time

from openai import OpenAI
import os
from google.colab import userdata

MODEL = "gpt-4o"

client = OpenAI(api_key=userdata.get('oaitest'))

import openai

api_key = userdata.get('oaitest')
openai.api_key = api_key

"""# Data Acquisition"""

drive.mount('/content/drive')

base_dir = '/content/drive/My Drive/clickbait_data'

file_path = '/content/drive/My Drive/clickbait_data/clickbait_data.csv'

df = pd.read_csv(file_path)

df.head()

num_rows = len(df)
print(f"The DataFrame has {num_rows} rows.")

df_copy = df.copy()

# sensational analysis on df_copy
sensational_results = []

start_time = time.time()

# batch API requests and add delays between each request
batch_size = 10
delay = 5  # 5secs delay

for i in range(0, len(df_copy), batch_size):
    batch_headline = df_copy['headline'][i:i+batch_size]

    for text in batch_headline:
        # prompt
        messages = [
            {"role": "system", "content": "You are a Chief Natural Language Processing and Linguistics Engineer."},
            {"role": "user", "content": f"""
                Analyze the sensational level of the following text: {text}.
                Sensational score instruction:
                (0) Not at all (mean 0–0.75)
                (1) Not too much (mean 0.76–1.50)
                (2) Somewhat (mean 1.51–2.25)
                (3) Fairly (mean 2.26–3.25)
                (4) Very (mean 3.26–4)
                Reply in three columns: Sensational or Not (Yes or No), Sensational Scores (0-4), Reason.
                Reply only Yes or No, only Sensational Scores(0-4), only Reason. Do not quote the original text. Do not use any word such as 'here's blahblahblah...'. Only focus on the answers.
                Use \ to seperate three columns.
                Sensational Scores should be a number(to two decimal places) instead of a range. Just score as the highest score of that range.
                Sensational Scores don't need always be end as 0 or 5. The two decimal places could be 0 to 9.
            """}
        ]
        try:
            # GPT-4 sensational
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
            model=MODEL,
            messages=messages
            )

            # save sensational result
            analysis_result = response.choices[0].message.content.strip()
            sensational_results.append(analysis_result)
        except Exception as e:
            print(f"An error occurred: {e}")
            # record error line for late processing
            sensational_results.append(analysis_result)

    # add delay between each batch
    time.sleep(delay)

# add result to DataFrame copy
df_copy['sensational_analysis'] = sensational_results

end_time = time.time()

# execution time
execution_time = end_time - start_time
print(f"Total execution time: {execution_time} seconds")

print(df_copy.head(10))

# dispay all content for each line
pd.set_option('display.max_colwidth', None)

print(df_copy.head(10))

# save path
csv_file_path = '/content/drive/My Drive/clickbait_data/sensational_results.csv'

# 7. save new dataFrame as CSV
df_copy.to_csv(csv_file_path, encoding='utf-8',index=False)

print(f"CSV file saved to {csv_file_path}")
