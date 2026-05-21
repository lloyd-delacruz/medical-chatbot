# Guide For Me — How to Rebuild the Physiotherapy Chatbot (in plain English)

This is a friendly, no-jargon walkthrough of what `trial_physio.ipynb` does and how
to rebuild it from scratch. Follow it top to bottom and you'll end up with a chatbot
that answers physiotherapy questions using an evidence-based physical therapy textbook
as its source.

---

## The big idea (read this first)

Think of this project like giving an AI an **open-book exam**.

- The "book" is `data/Evidence-physical-therapy-K.-Bo.pdf`.
- Instead of hoping the AI already memorized the answer, we let it **look up the
  most relevant pages first**, then write an answer based only on what it found.

That "look it up, then answer" trick is called **RAG** (Retrieval-Augmented
Generation). It makes answers more accurate and lets the AI cite a real source
instead of making things up.

To make "looking it up" fast, we do some prep work:
1. Read the PDF into text.
2. Chop the text into small passages ("chunks").
3. Turn each chunk into a list of numbers that captures its *meaning*
   ("embeddings").
4. Store those numbers in a specialized search engine ("vector database" =
   Pinecone).
5. When a question comes in, find the most similar chunks and hand them to the
   AI (GPT-4o) to write the final answer.

That's the whole project. Everything below is just doing those 5 things, one
small step at a time.

---

## Before you start (one-time setup)

1. **Use the right environment.** This project's packages live in the project's
   virtual environment **`.venv`** (Python 3.13) — *not* the default system Python.
   - In the notebook, set the kernel (top-right in VS Code, or Kernel menu in
     Jupyter) to **`Python (physio-chatbot .venv)`**.
   - If you don't see it, reload your editor window first. The kernel is registered
     from `.venv` via `python -m ipykernel install`.

2. **Get your API keys.** You need two accounts (both have free tiers to start):
   - **Pinecone** → for the vector database. Copy your API key.
   - **OpenAI** → for the GPT-4o brain. Copy your API key.
   > Note: OpenAI charges per use. The embedding step here is free (it runs on
   > your own machine), but the GPT-4o answers cost a small amount per question.

3. **Create a `.env` file** in the project root (`physio-chatbot/.env`) and put
   your keys in it. A `.env` file is just a safe place to store secrets so they
   aren't written directly in your code:
   ```env
   PINECONE_API_KEY=your_pinecone_key_here
   OPENAI_API_KEY=your_openai_key_here
   ```

4. **Make sure the PDF is there:** `physio-chatbot/data/Evidence-physical-therapy-K.-Bo.pdf`.

---

## Part 1 — Point the notebook at the project folder

The notebook lives in `research/`, but the data lives one level up in `data/`.
These first cells just move our "current location" up to the project root so the
path `"data"` works.

```python
print("OK")          # sanity check that the kernel runs
```
```python
%pwd                 # "print working directory" — shows where we are now
```
```python
import os

# Move to the project root so paths like "data" and ".env" resolve.
# Idempotent: only steps up while still inside research/, so re-running is safe.
while os.path.basename(os.getcwd()) == "research":
    os.chdir("..")
print("Working dir:", os.getcwd())
```

**Why:** so that later, `load_physio_pdf("data")` looks in the real `data/`
folder instead of a folder that doesn't exist inside `research/`. The loop makes
the cell safe to re-run — a plain `os.chdir("../")` would walk *up another level*
every time you ran it.

---

## Part 2 — Read the PDF into text

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

```python
# Extract text from ONLY the physiotherapy PDF
def load_physio_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="Evidence-physical-therapy-K.-Bo.pdf",  # this specific book only
        loader_cls=PyPDFLoader                        # use the PDF reader for each file
    )
    return loader.load()
```
```python
extracted_data = load_physio_pdf("data")
len(extracted_data)     # how many pages were loaded (~435)
```

**What you get:** a list where each item is one page of the book, with its text
and some metadata (like which file it came from).

---

## Part 3 — Trim the data down to the essentials

Each page comes with lots of metadata we don't need. This step keeps only the
text and the source filename, so things stay tidy and small.

```python
from typing import List
from langchain_core.documents import Document

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only the page text and its 'source' filename."""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs
```
```python
minimal_docs = filter_to_minimal_docs(extracted_data)
```

**Why:** less clutter = cleaner search results later.

---

## Part 4 — Chop the text into small chunks

A whole page is too big to search precisely. We cut the text into ~500-character
passages that slightly overlap, so we don't accidentally split a sentence's
meaning in half.

```python
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # each chunk ~500 characters
        chunk_overlap=20,    # repeat 20 chars between chunks for continuity
    )
    return text_splitter.split_documents(minimal_docs)
```
```python
texts_chunk = text_split(minimal_docs)
print(f"Number of chunks: {len(texts_chunk)}")   # ~3838
```

**Analogy:** instead of searching "page 212," we can now search "this exact
paragraph about exercise prescription." Much more precise.

---

## Part 5 — Turn text into "meaning numbers" (embeddings)

An **embedding** converts a piece of text into a list of numbers (here, 384 of
them) that captures its *meaning*. Texts about similar topics get similar
numbers. This is what makes "find me passages like this question" possible.

This model runs **on your own computer** (free, no API needed). The first run
downloads the model, so it may take a minute.

```python
from langchain_huggingface import HuggingFaceEmbeddings

def download_embeddings():
    """Download and return the HuggingFace embeddings model (384-dim)."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

embedding = download_embeddings()
```
```python
# Quick test: turn one sentence into numbers
vector = embedding.embed_query("Hello world")
print("Vector length:", len(vector))   # should print 384
```

**Remember the number 384** — the vector database needs to know each item is a
list of 384 numbers.

---

## Part 6 — Load your secret keys

This reads the API keys from your `.env` file so the code can talk to Pinecone
and OpenAI without hard-coding secrets.

```python
from dotenv import load_dotenv
import os

# load_dotenv() reads the .env at the project root and populates os.environ.
# In a notebook it searches the current working directory, so run the Part 1
# cell first. It returns True when a .env was found and loaded.
loaded = load_dotenv()
print("Loaded .env:", loaded)
```
```python
# Read the keys and fail with a clear message if any are missing
# (instead of a cryptic "TypeError: str expected, not NoneType").
required = ["PINECONE_API_KEY", "OPENAI_API_KEY"]
missing = [name for name in required if not os.getenv(name)]
if missing:
    raise RuntimeError(
        f"Missing {missing} in the environment. Add them to the .env file at the "
        "project root, then re-run the cell above."
    )

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
print("API keys loaded ✓")
```

> If `Loaded .env:` prints `False`, your `.env` file is missing, empty, or in the
> wrong folder (it must be in the project root, not `research/`).

---

## Part 7 — Connect to Pinecone

**Pinecone** is a cloud "search engine for meaning numbers."

```python
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
```

---

## Part 8 — Create the index

An **index** is just the named container that holds the embeddings.

```python
from pinecone import ServerlessSpec

index_name = "physio-chatbot"

if not pc.has_index(index_name):           # only create it if it doesn't exist
    pc.create_index(
        name=index_name,
        dimension=384,                     # must match our embedding size!
        metric="cosine",                   # how it measures "similarity"
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
```

**Key point:** `dimension=384` must match the embedding length from Part 5. If
they don't match, Pinecone will reject the data.

---

## Part 9 — Upload your chunks to Pinecone

This takes every chunk, turns it into numbers (using the embedding model), and
stores it in Pinecone. **Run this once** to fill the database.

```python
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name,
)
```

After it's been uploaded once, you don't need to re-upload. On later runs, just
**connect to the existing index** instead:

```python
# Use this on future runs instead of re-uploading everything
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding,
)
```

> Tip: Run **Part 9's first block once**, then use the "existing index" block from
> then on. Otherwise you re-upload every time.

---

## Part 10 — Build the "retriever" (the look-up part)

The **retriever** is the piece that, given a question, fetches the most relevant
chunks. `k=3` means "give me the top 3 closest matches."

```python
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
```
```python
# Test it: what does it find for this question?
retrieved_docs = retriever.invoke("What is evidence-based physical therapy?")
retrieved_docs
```

---

## Part 11 — Connect the AI brain (GPT-4o)

```python
from langchain_openai import ChatOpenAI
chatModel = ChatOpenAI(model="gpt-4o")
```

This is the part that actually writes answers in natural language. (This is the
step that costs a little money per question.)

---

## Part 12 — Wire it all together (the RAG chain)

Now we glue the retriever and the AI together with an instruction ("system
prompt") that tells the AI how to behave: answer using only the retrieved
context, stay concise, and admit when it doesn't know.

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
```
```python
system_prompt = (
    "You are a Physiotherapy assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"          # the retrieved chunks get inserted here
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),   # the user's question goes here
])
```
```python
# 1) chain that stuffs the chunks + question into the AI
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
# 2) chain that first retrieves chunks, then runs the above
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

**In plain terms:** `rag_chain` = "look up the 3 best chunks, paste them into the
instructions, then ask GPT-4o to answer."

---

## Part 13 — Ask it questions!

```python
response = rag_chain.invoke({"input": "What is evidence-based physical therapy?"})
print(response["answer"])
```
```python
response = rag_chain.invoke({"input": "What can randomized trials and systematic reviews tell us?"})
print(response["answer"])
```
```python
response = rag_chain.invoke({"input": "How should a physiotherapist appraise clinical evidence?"})
print(response["answer"])
```

If you get sensible physiotherapy answers here — congratulations, the chatbot works. 🎉

---

## Quick mental model (cheat sheet)

| Step | Plain-English job | Tool used |
|------|-------------------|-----------|
| Load PDF | Read the book into text | PyPDFLoader |
| Filter | Keep only what matters | langchain_core Document |
| Chunk | Cut into searchable passages | RecursiveCharacterTextSplitter |
| Embed | Turn text into meaning-numbers | HuggingFace MiniLM (384 dims) |
| Store | Save numbers in a search engine | Pinecone |
| Retrieve | Find the closest passages | retriever (k=3) |
| Generate | Write the final answer | GPT-4o |
| Chain | Glue retrieve + generate together | RAG chain |

---

## If something breaks

- **`ModuleNotFoundError` / `ImportError: pypdf not found`** → wrong kernel.
  Select **`Python (physio-chatbot .venv)`** and re-run.
- **`Loaded .env: False` / API key missing** → `.env` missing/empty or in the
  wrong folder (must be the project root, not `research/`).
- **Pinecone dimension error** → the index `dimension` (384) must match the
  embedding length.
- **Path/"file not found" for the PDF** → make sure you ran the Part 1 cell so
  you're in the project root, and that
  `data/Evidence-physical-therapy-K.-Bo.pdf` exists.
