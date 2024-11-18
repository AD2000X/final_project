!pip install openai --quiet

from google.colab import drive
import glob
import os
import json
import pandas as pd

from openai import OpenAI
from google.colab import userdata

MODEL = "gpt-4o"

client = OpenAI(api_key=userdata.get('oaitest'))

import openai

api_key = userdata.get('oaitest')
openai.api_key = api_key

drive.mount('/content/drive', force_remount=True)

base_dir = '/content/drive/My Drive/arousal'

file_path = f'{base_dir}/clickbait_data_arousal.csv'

encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'utf-16']
for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f'Successfully read the file with encoding: {encoding}')
        print(df.head())
        break
    except UnicodeDecodeError as e:
        print(f'Failed to read the file with encoding: {encoding}')
        print(f'Error: {e}')

df.head()

num_rows = len(df)
print(f"The DataFrame has {num_rows} rows.")

import time
import logging

# set log
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

df_copy = df.copy()

# arousal on de_copy
arousal_results = []

start_time = time.time()

# Batch API requests and add delays between each request
batch_size = 20
delay = 5  # 5 secs delay

for i in range(0, len(df_copy), batch_size):
    batch_headline = df_copy['headline'][i:i+batch_size]

    for text in batch_headline:
        # prompt
        messages = [
            {"role": "system", "content": "You are a Chief Natural Language Processing and Linguistics Engineer."},
            {"role": "user", "content": f"""
                Analyze the arousal level of the following text: {text}.
                Task1: Identify the emotion of each news headline.
                Emotion could be neutral.
                Eight Emotions categories: anger, anticipation, disgust, fear, joy, sadness, surprise, and trust.
                Task 2: Score for the Arousal level of each news headlines. Combine all the evaluate criteria into a proper single score.
                Arousal difenition: This dimension measures the level of physiological and psychological activation.
                It ranges from low arousal (calm, relaxed) to high arousal (excited, agitated).
                For instance, emotions such as calmness or boredom are low in arousal, while excitement and fear are high in arousal.
                Arousal score instruction:
                (0) Not at all (0-0.353)
                (1) Not too much (0.354-0.439)
                (2) Somewhat (0.440-0.521)
                (3) Fairly (0.522-0.647)
                (4) Very (0.648-1.000)
		            Reply instruction:
                Reply in four columns:
                1. Emotion(s) or neutral.
                2. Arousal or Not (Yes or No).
                3. Arousal Scores for the news headline (0-4).
                4. Reason for the score of the arousal of news headline, not for all the emotions.
                Use // to seperate these five columns.
                Stricktly reply all emotion(s) or neutral in one column.
                Stricktly reply reasons only for arousal score, do not reply for all reasons for the emotions.
                Reply arousal only Yes or No. Reply only Arousal Scores (0.000-1.000). Do not quote the original text. Do not use any word such as 'here's blahblahblah...', only focus on the answers.
                Do not repeat 'Emotion(s) or neutral // Arousal // Arousal Scores // Reason for the score' in all replies.
                Arousal Scores should be a number(to three decimal places) instead of a range. Just score as the highest score of that range.
                Arousal Scores don't need always be end as 0 or 5. The three decimal places could be 0 to 9.
                Correct reply example (do not repeat this line in all replies) is one reply for one news headline: joy, anticipation, surprise, trust // Yes // 0.700 // The headline mentions a substantial monetary pledge to a medical center, which can be exciting and noteworthy, leading to higher arousal levels.)
            """}
        ]
        try:
            # arousal by GPT-4
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
            model=MODEL,
            messages=messages
            )

            # save arousal result
            analysis_result = response.choices[0].message.content.strip()
            arousal_results.append(analysis_result)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            print(f"An error occurred: {e}")
            # record error line for later handle
            arousal_results.append("Error")

    # add delay between each batch
    time.sleep(delay)

# save results to DataFrame copy
df_copy['arousal_analysis'] = arousal_results


end_time = time.time()

# execution time
execution_time = end_time - start_time
print(f"Total execution time: {execution_time} seconds")

print(df_copy.head(10))

# save path
csv_file_path = '/content/drive/My Drive/arousal/clickbait_arousal_results.csv'

# 7. save new DataFrame as CSV
df_copy.to_csv(csv_file_path, encoding='utf-8',index=False)

print(f"CSV file saved to {csv_file_path}")
