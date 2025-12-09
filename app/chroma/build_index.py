from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import pandas as pd

load_dotenv()
client = OpenAI()

print("시작")

def embed(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return resp.data[0].embedding

chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection(
    name="empathy_training",
    metadata={"hnsw:space": "cosine"},
)

df = pd.read_excel("empathy_training_500.xlsx")

for _, row in df.iterrows():
    text = row["공감 멘트"]
    label = row["공감 유형"]

    print(row)

    vector = embed(text)

    collection.add(
        ids=[str(row["ID"])],
        embeddings=[vector],
        metadatas=[{"label": label}],
        documents=[text],
    )

print("✅ 인덱싱 완료!")
