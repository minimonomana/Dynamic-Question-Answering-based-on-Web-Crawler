# !pip install transformers sentence-transformers

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load your data
df = pd.read_csv('processed/shortened.csv')

# Load pre-trained SentenceTransformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for your dataset
df['embeddings'] = df['text'].apply(lambda x: st_model.encode(x, convert_to_tensor=True))

# Calculate the number of tokens for each text
df['n_tokens'] = df['text'].apply(lambda x: len(x.split()))

# Save the embeddings
df.to_pickle('processed/embeddings.pkl')
df.head()

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

def create_context(question, df, max_len=1800):
    # Get the embeddings for the question
    q_embeddings = st_model.encode(question, convert_to_tensor=True)

    # Compute cosine similarities
    df['distances'] = df['embeddings'].apply(lambda x: cosine_similarity([q_embeddings], [x]).flatten()[0])

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=False).iterrows():
        cur_len += row['n_tokens'] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)

def answer_question(df, question, max_len=1800, max_new_tokens=150):
    context = create_context(question, df, max_len=max_len)
    input_text = f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
    outputs = model.generate(
        inputs.input_ids,
        max_length=inputs.input_ids.shape[1] + max_new_tokens,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer.split('Answer:')[-1].strip()


# Test the function
print(answer_question(df, question="Version control systems (VCSs) are tools used to track changes to source code (or other collections of files and folders). What does these tools help?"))
print(answer_question(df, question="Why is version control useful?"))
print(answer_question(df, question="What is the challenge of Asymmetric-key cryptography?"))