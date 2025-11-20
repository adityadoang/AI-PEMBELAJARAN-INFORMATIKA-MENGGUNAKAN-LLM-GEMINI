import mysql.connector
import json
import ollama
from sentence_transformers import SentenceTransformer

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:1.5b"

llm_agent = ollama.Client(host=OLLAMA_HOST)
embedder = SentenceTransformer('BAAI/bge-m3')

db = mysql.connector.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="2j5p3FAFY9SMKvS.root",
    password="thbpnJzHhVwYy0bj",
    database="RAG",
    ssl_ca=r"E:\kulish\smt3\kb\CA\isrgrootx1.pem",
    ssl_verify_cert=True,
    ssl_verify_identity=True
)

def search_documents(database, query, k_top=5):
    results = []
    query_embedding_list = embedder.encode(query).tolist()
    # Jika fungsi vektor Anda menerima JSON array sebagai string, ini OK
    query_embedding_str = json.dumps(query_embedding_list)

    curr = database.cursor()
    sql_query = f"""
        SELECT text, vec_cosine_distance(embedding, %s) AS distance
        FROM documents
        ORDER BY distance ASC
        LIMIT {k_top}
    """
    curr.execute(sql_query, (query_embedding_str,))
    rows = curr.fetchall()
    curr.close()  # tidak perlu commit untuk SELECT

    for row in rows:
        text, distance = row
        results.append({"text": text, "distance": distance})
    return results

def response_query(database, query, conversation_history):
    retrieved_docs = search_documents(database, query)
    context = "\n".join([doc['text'] for doc in retrieved_docs])
    
    # Tambahkan riwayat percakapan ke prompt
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    prompt = f"Conversation history:\n{history_text}\n\nAnswer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Sertakan riwayat dalam messages untuk Ollama
    messages = conversation_history + [{"role": "user", "content": prompt}]
    response = llm_agent.chat(model=OLLAMA_MODEL, messages=messages)
    
    return response['message']['content']

if __name__ == "__main__":
    print("MAU NANYA APA NIH")
    conversation_history = []  # List untuk menyimpan riwayat percakapan
    
    while True:
        query_text = input("You: ")

        if query_text.lower() in ['exit', 'quit', 'q']:
            print("Exiting chat. Goodbye!")
            break

        # Tambahkan query pengguna ke riwayat
        conversation_history.append({"role": "user", "content": query_text})
        
        response = response_query(database=db, query=query_text, conversation_history=conversation_history)
        
        # Tambahkan respons bot ke riwayat
        conversation_history.append({"role": "assistant", "content": response})
        
        print("Bot:", response)

print("SELESAI")
