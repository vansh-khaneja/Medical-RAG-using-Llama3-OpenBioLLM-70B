from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
set_seed(1234)

df = pd.read_csv('train.csv')

ques_data = df['Question'].tolist()[:100]
answer_data = df['Answer'].tolist()[:100]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
vectors = model.encode(ques_data)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(":memory:")

client.recreate_collection(
    collection_name="doc_data",
    vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
)

client.upload_collection(
    collection_name="doc_data",
    ids=[i for i in range(len(ques_data))],
    vectors=vectors,
)

def fetch_best_match(question):
  model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
  ques_vector = model.encode(question)
  result = client.query_points(collection_name="doc_data",query=ques_vector)
  sim_ids = []
  for i in result.points:
    sim_ids.append(i.id)
  context = answer_data[sim_ids[0]]
  return context


def fetch_llm_response(question,context):
  model_checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
  model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="cuda")
  chat = [
    {"role": "user", "content": f"this is question {question} asked by user you are a chatbot answer the question based on this context {context} in not more than 3-4 points"}
  ]
  token_inputs = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt", add_generation_prompt=True).to("cuda")
  token_outputs = model.generate(input_ids=token_inputs, do_sample=True, max_new_tokens=500, temperature=.5)
  new_tokens = token_outputs[0][token_inputs.shape[-1]:]
  decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
  return decoded_output


import gradio as gr

def chatbot_response(user_input):
    if user_input:
      context = fetch_best_match(user_input)
      return fetch_llm_response(user_input,context)
    else:
        return "Bot: Please enter a message."

iface = gr.Interface(
    fn=chatbot_response,
    inputs="text",
    outputs="text",
    title="Medical RAG",
    description="Type your issue below and the bot will respond."
)

iface.launch()
