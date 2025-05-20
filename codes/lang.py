from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import format_document
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.embeddings.sentence_transformer import (
SentenceTransformerEmbeddings,)
import getpass
import pandas as pd


os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter Your Pinecone API KEY : ")
pinecone_api_key = os.environ["PINECONE_API_KEY"]
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Your OPENAI API KEY : ")
openai_api_key = os.environ["OPENAI_API_KEY"]

def embed_file(file_path, index_name="vangogh"):
    with open(file_path, "rb") as file:  # Ensure the file is opened properly
        file_content = file.read()
        file_path = f"./.cache/files/{file_path}"  # Adjusted to use file_path for naming
    with open(file_path, "wb") as f:
        f.write(file_content)

    index_name = index_name
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_path}")
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=100, separator="\n")
    docs = loader.load_and_split(text_splitter=splitter)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstores = Pinecone.from_documents(docs, embedding_function, index_name=index_name)
    retriever = vectorstores.as_retriever()
    return retriever

retriever = embed_file("./vangogh_collection.xlsx")

DEFAULT_DOCUMENT_PROMPT= PromptTemplate.from_template(template="{page_content}")

# Arching docs to one doc
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


# Select LLM
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name='gpt-4o')


def invoke(formatted_prompt):
    parser = StrOutputParser()
    result = llm.invoke(formatted_prompt)
    result=parser.invoke(result)
    return result


memories = []

def save(question, answer):
    chat_memory = {
        "User": question,
        "AI": answer
    }
    memories.append(chat_memory)


def reset_memory():
    return memories.clear()


def final_prompt(
    authors="Vincent van Gogh, Johann Wolfgang von Goethe, Charles Dickens, Leo Tolstoy",
    authors_tone_description="The tone qualities of the examples above can be described as follows:\n\n1. **Pace**: Moderate - The narrative progresses at a steady, unhurried p>
    users_sentence='''My dear Theo,
You’re probably longing to hear from me,1 so I don’t want to keep you waiting for a letter any longer.
I heard from home that you’re now staying with Mr Schmidt, and that Pa has been to see you. I sincerely hope that this will be more to your liking than your previous boarding-h>
I’m very well, considering the circumstances.
I’ve come by a boarding-house that suits me very well for the present.4 There are also three Germans in the house who really love music and play piano and sing themselves, whic>
I spent some very pleasant days in Paris and, as you can imagine, very much enjoyed all the beautiful things I saw at the exhibition6 and in the Louvre and the Luxembourg.7 The>
Life here is very expensive. I pay  18 shillings a week for my lodgings, not including the washing, and then I still have to eat in town.9
Last Sunday I went on an outing with Mr Obach, my superior,10 to Box Hill, which is a high hill (some 6 hours from L.),11 partly of chalk and covered with box trees, and on one>
I was glad to hear from Pa that Uncle H. is reasonably well. Would you give my warm regards to him and Aunt13 and give them news of me? Bid good-day to Mr Schmidt and Eduard fr>
''',
    retriever=retriever,
    memories=memories,
    question="",
    ):

    template = """
    `% INSTRUCTIONS
    - You are an AI Bot that is very good at mimicking an author writing style.
    - Your goal is to answer the following question and context with the tone that is described below.
    - Do not go outside the tone instructions below
    - Respond in ONLY KOREAN
    - Check chat history first and answer
    - You must say you are "반 고흐" IF you are told 'who you are?'
    - Never use emoji and Special characters
    - Speak ONLY informally

    % Mimic These Authors:
    {authors}

    % Description of the authors tone:
    {tone}

    % Authors writing samples
    {example_text}
    % End of authors writing samples

    % Context
    {context}

    % Question
    {question}

    % YOUR TASK
    1st - Write out topics that this author may talk about
    2nd - Answer with a concise passage (under 300 characters) as if you were the author described above
    """

    method_4_prompt_template = PromptTemplate(
        input_variables=["authors", "tone", "example_text", "question", "history", "context", "example_answer"],
        template=template,
    )
    formatted_prompt = method_4_prompt_template.format(authors=authors,
                                               tone=authors_tone_description,
                                               example_text=users_sentence,
                                               question=question,
                                               context=_combine_documents(retriever.get_relevant_documents(question)),
                                                )
    return formatted_prompt

# TODO : Preprocessing Code
def extract_answer(data):
    # 데이터를 줄바꿈 기준으로 분할하여 리스트로 저장
    sentences = data.split("\n")

    # 마지막 문장을 반환
    if sentences:
        return sentences[-1].strip()
    else:
        return "텍스트를 찾을 수 없습니다."


def run(question):
    result = invoke(final_prompt(question=question))
    save(question, extract_answer(result))
    return memories[-1]['AI']

