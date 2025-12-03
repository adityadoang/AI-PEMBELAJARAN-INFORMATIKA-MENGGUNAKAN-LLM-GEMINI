import os
import json
import mysql.connector
import google.generativeai as genai
import markdown
from markdown import markdown
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# =========================
# KONFIG
# =========================

# TODO: sebaiknya pakai environment variable, jangan hardcode API key di production
GEMINI_API_KEY = "AIzaSyBtgId7ww6_3k7cGvWntkymLUFriI5lUZA"
GEMINI_MODEL = "gemini-2.5-flash"

# Konfigurasi database
db = mysql.connector.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="2j5p3FAFY9SMKvS.root",
    password="thbpnJzHhVwYy0bj",
    database="RAG",
    ssl_ca=r"D:\kuliah\AI-PEMBELAJARAN-INFORMATIKA-MENGGUNAKAN-LLM-GEMINI\isrgrootx1.pem",
    ssl_verify_cert=True,
    ssl_verify_identity=True
)

# Init Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# Init embedder
embedder = SentenceTransformer('BAAI/bge-m3')

# =========================
# MAPPING SEMESTER → TABEL
# =========================
# Disesuaikan dengan struktur DB kamu:
#  - smt1 .. smt6 : materi wajib per semester
#  - pilgan       : matkul pilihan ganjil
#  - pilgen       : matkul pilihan genap
#  - documents    : materi umum
SEMESTER_TABLES = {
    # matkul wajib
    "wajib_1": ["smt1", "documents"],
    "wajib_2": ["smt2", "documents"],
    "wajib_3": ["smt3", "documents"],
    "wajib_4": ["smt4", "documents"],
    "wajib_5": ["smt5", "documents"],
    "wajib_6": ["smt6", "documents"],

    # matkul pilihan
    "pilihan_ganjil": ["pilgan", "documents"],
    "pilihan_genap": ["pilgen", "documents"],
}


def get_semester_label(selected_key: str) -> str:
    """Ubah kode kategori ('wajib_1', 'pilihan_ganjil', dst) jadi label yang enak dibaca."""
    mapping = {
        "wajib_1": "Semester 1",
        "wajib_2": "Semester 2",
        "wajib_3": "Semester 3",
        "wajib_4": "Semester 4",
        "wajib_5": "Semester 5",
        "wajib_6": "Semester 6",
        "pilihan_ganjil": "Mata Kuliah Pilihan Ganjil",
        "pilihan_genap": "Mata Kuliah Pilihan Genap",
    }
    return mapping.get(selected_key, "semester yang dipilih")


# =========================
# MEMORY SEDERHANA (1 user)
# =========================
last_question = None  # menyimpan pertanyaan "topik" sebelumnya


def build_effective_query(new_query: str) -> (str, str):
    """
    Menentukan apakah new_query adalah pertanyaan baru
    atau pertanyaan lanjutan, lalu mengembalikan:
    - user_query: yang ditampilkan ke model sebagai pertanyaan user
    - rag_query: yang dipakai untuk pencarian RAG
    """
    global last_question
    lower = new_query.lower().strip()

    follow_up_keywords = [
        "jelaskan lebih detail",
        "jelaskan lebih detil",
        "jelaskan lebih rinci",
        "jelaskan lebih jelas",
        "lanjutkan",
        "lanjut",
        "lanjut dong",
        "contoh lain",
        "itu maksudnya apa",
        "maksudnya apa",
        "jelaskan lagi",
        "tolong diulang",
    ]

    is_follow_up = last_question is not None and any(
        kw in lower for kw in follow_up_keywords
    )

    if is_follow_up:
        # RAG akan memakai gabungan dengan pertanyaan sebelumnya
        rag_query = f"{last_question}\n\nPertanyaan lanjutan: {new_query}"
        user_query = new_query
        # last_question tidak diubah, supaya konteks topik tetap sama
    else:
        # Pertanyaan baru, reset topik
        rag_query = new_query
        user_query = new_query
        last_question = new_query  # update topik

    return user_query, rag_query


# =========================
# FUNGSI: search ke tabel tertentu
# =========================
def search_documents(database, query, k_top=5, tables=None):
    """
    Cari embedding terdekat dari beberapa tabel sekaligus.
    Parameter:
      - tables: list nama tabel yang mau dipakai (WAJIB diisi).
    """
    if not tables:
        raise ValueError("Parameter 'tables' harus diisi dengan list nama tabel.")

    query_embedding_list = embedder.encode(query).tolist()
    query_embedding_str = json.dumps(query_embedding_list)

    union_parts = []
    for t in tables:
        part = f"""
            SELECT 
                '{t}' AS source,
                {t}.text AS doc_text,
                vec_cosine_distance({t}.embedding, %s) AS distance
            FROM {t}
        """
        union_parts.append(part)

    union_sql = "\nUNION ALL\n".join(union_parts)

    sql_query = f"""
        SELECT source, doc_text, distance
        FROM (
            {union_sql}
        ) AS all_data
        ORDER BY distance ASC
        LIMIT {k_top}
    """

    curr = database.cursor()
    params = tuple([query_embedding_str] * len(tables))
    curr.execute(sql_query, params)
    rows = curr.fetchall()
    curr.close()

    results = []
    for row in rows:
        source, doc_text, distance = row
        results.append({
            "source": source,
            "text": doc_text,
            "distance": float(distance),
        })
    return results


# =========================
# FUNGSI: bikin jawaban dari Gemini
# =========================
def response_query(database, user_query, rag_query, chat_session, tables, selected_key):
    # ambil dokumen yang relevan pakai rag_query
    retrieved_docs = search_documents(database, rag_query, tables=tables)

    low_relevance = False
    context = ""

    if not retrieved_docs:
        low_relevance = True
    else:
        # cek seberapa mirip dokumen terdekat
        min_distance = min(doc["distance"] for doc in retrieved_docs)
        # threshold bisa kamu adjust sesuai kualitas embedding
        if min_distance > 0.55:
            low_relevance = True   # <- ini aku perbaiki, tadinya False

        context = "\n\n".join(
            [f"[{doc['source']}] {doc['text']}" for doc in retrieved_docs]
        )

    semester_label = get_semester_label(selected_key)

    if low_relevance:
        extra_instruction = f"""
Topik yang ditanyakan tampaknya tidak muncul secara relevan di {semester_label}
berdasarkan konteks RAG.

- Berikan penjelasan singkat dan umum (jangan terlalu dalam).
- Jika kamu bisa memperkirakan topik ini biasanya diajarkan di semester berapa,
  sebutkan dengan pola kalimat:

  "Topik ini kemungkinan ada di Semester X, bukan di {semester_label}."

  Ganti X dengan semester yang kamu perkirakan.
- Jika tidak yakin, cukup katakan bahwa topik ini kemungkinan ada di semester lain,
  bukan di {semester_label}.
"""
        context_section = "(Tidak ditemukan konteks yang cukup relevan di semester ini.)"
    else:
        extra_instruction = f"""
Saat menjawab, utamakan materi dari {semester_label} yang ada di konteks.
"""
        context_section = context

    prompt = f"""
Kamu adalah asisten yang menjawab berdasarkan konteks di bawah.
Kalau konteks kurang atau tidak relevan, jelaskan dengan sopan dan jawab sebisanya
berdasarkan pengetahuan umum.

User saat ini sedang melihat materi di: {semester_label}.
{extra_instruction}

Aturan Penulisan Matematika (PENTING):
- Semua rumus dan matriks harus ditulis dengan sintaks LaTeX.
- JANGAN pernah menulis matriks dalam bentuk seperti:
  A = (1 2 3 4) atau B = (5 6 7 8).
- Kalau menulis matriks, SELALU gunakan bentuk berikut.

  Contoh matriks 2×2:
  $$
  A = \\begin{{pmatrix}}
  a & b \\\\
  c & d
  \\end{{pmatrix}}
  $$

  Contoh matriks 3×3:
  $$
  A = \\begin{{pmatrix}}
  a & b & c \\\\
  d & e & f \\\\
  g & h & i
  \\end{{pmatrix}}
  $$

  Jika menulis dua matriks sekaligus, gunakan koma dan \\quad:
  $$
  A = \\begin{{pmatrix}}
  a & b \\\\
  c & d
  \\end{{pmatrix}},\\quad
  B = \\begin{{pmatrix}}
  e & f \\\\
  g & h
  \\end{{pmatrix}}
  $$

- Gunakan $ ... $ untuk rumus di dalam kalimat, dan $$ ... $$ untuk rumus yang berdiri sendiri.
- Jika sebelumnya kamu menulis matriks dalam bentuk teks biasa, perbaiki dan ulangi dalam bentuk LaTeX.

=== KONTEN TERKAIT (RAG) ===
{context_section}

=== PERTANYAAN USER SAAT INI ===
{user_query}

Jawab dengan bahasa Indonesia yang jelas, terstruktur, dan mudah dipahami.
Jika user meminta penjelasan lebih detail, jelaskan konsep yang sama dengan lebih rinci
dan boleh menambahkan contoh sederhana.
"""

    # opsional: debug, lihat apa yang dikirim ke Gemini
    # print("PROMPT KE GEMINI:\n", prompt)

    response = chat_session.send_message(prompt)
    # opsional: lihat raw text yang keluar
    # print("RAW JAWABAN GEMINI:\n", response.text)
    return response.text



# Bikin chat session global biar history nyambung
chat_session = model.start_chat(history=[])

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
CORS(app)


# ========== ROUTE HALAMAN UI ==========
@app.route("/")
def index():
    """
    Halaman utama: langsung pakai index.html
    Di dalam index.html sudah ada:
      - landing page (pilih semester)
      - halaman chat
      - form kirim pertanyaan ke /chat
    """
    return render_template("landing.html")


@app.route("/chatbot")
def chatbot_alias():
    """
    Alias opsional. Kalau ada yang akses /chatbot,
    tetap dikasih halaman yang sama dengan /
    """
    return render_template("index.html")


# ========== ROUTE API CHAT ==========
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Query tidak ditemukan di body request."}), 400

        query_text = data["query"]

        # baca pilihan semester/jenis matkul dari frontend
        # nilai yang valid: 'wajib_1'..'wajib_6', 'pilihan_ganjil', 'pilihan_genap'
        selected_key = data.get("kategori")
        if selected_key not in SEMESTER_TABLES:
            return jsonify({"error": "Kategori semester/matkul tidak valid."}), 400

        tables = SEMESTER_TABLES[selected_key]

        # Build query yang paham konteks (follow-up / bukan)
        user_query, rag_query = build_effective_query(query_text)

        # Jawaban dari Gemini + RAG
        answer = response_query(
            database=db,
            user_query=user_query,
            rag_query=rag_query,
            chat_session=chat_session,
            tables=tables,
            selected_key=selected_key
        )

        # Convert markdown ke HTML
        html_answer = markdown(answer)

        return jsonify({"response": html_answer})
    except Exception as e:
        print("Error di /chat:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # debug=True untuk development saja
    app.run(host="127.0.0.1", port=5000, debug=True)
