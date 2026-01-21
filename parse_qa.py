import pandas as pd

df = pd.read_csv("question_answer.csv")
df['Answer'] = df['Answer'].str.replace(r'\[cite: \d+\]', '', regex=True)
df.to_csv("question_answer.csv", index=False)