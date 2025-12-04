import json
import pandas as pd
import mysql.connector

from sentence_transformers import SentenceTransformer

#bikin instance embedder
embedder = SentenceTransformer('BAAI/bge-m3')

import mysql.connector

db = mysql.connector.connect(
  host = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
  port = 4000,
  user = "2j5p3FAFY9SMKvS.root",
  password = "thbpnJzHhVwYy0bj",
  database = "RAG",
  ssl_ca = "E:\kulish\smt3\kb\CA\isrgrootx1.pem",
  ssl_verify_cert = True,
  ssl_verify_identity = True
)

curr =db.cursor()


df = pd.read_csv('smt2.csv')
print(df)


for index, row in df.iterrows():
    text=str(row['question'])+" "+str(row['answer'])
    
    try:
            embedding_list = embedder.encode(text).tolist()
            embedding_str = json.dumps(embedding_list)
            
            sql_query = """
                            INSERT INTO smt2(text,embedding)VALUES (%s, %s)
                """
            
            curr.execute(sql_query, (text,embedding_str))
            print(f"data index--{index} berhasil di tambah")
    except Exception as e :
            print(f"ERROR: {e}")
            print(f"data index--{index} berhasil di tambah")
            
db.commit()
curr.close()
print("Data berhasil di tambahkan ke database.")