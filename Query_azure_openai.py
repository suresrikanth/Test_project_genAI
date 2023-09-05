import os
import urllib
import requests
import random
import json
from collections import OrderedDict
from IPython.display import display, HTML, Markdown
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings import OpenAIEmbeddings
import datetime
from Query_azure_cg_search_ai import content

from common.prompts import COMBINE_QUESTION_PROMPT, COMBINE_PROMPT, COMBINE_PROMPT_TEMPLATE
from common.utils import (
    get_search_results,
    model_tokens_limit,
    num_tokens_from_docs,
    num_tokens_from_string
)

from dotenv import load_dotenv
load_dotenv("credentials.env")

# Setup the Payloads header
headers = {'Content-Type': 'application/json','api-key': os.environ['AZURE_SEARCH_KEY']}
params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

# Set the ENV variables that Langchain needs to connect to Azure OpenAI
os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
os.environ["OPENAI_API_TYPE"] = "azure"

MODEL = "gpt-35-turbo" # options: gpt-35-turbo, gpt-35-turbo-16k, gpt-4, gpt-4-32k
COMPLETION_TOKENS = 1000
llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0, max_tokens=COMPLETION_TOKENS)

QUESTION = "What is CLP?"

# Now we create a simple prompt template
prompt = PromptTemplate(
    input_variables=["question", "language"],
    template='Answer the following question: "{question}". Give your response in {language}',
)

print(prompt.format(question=QUESTION, language="English"))

# And finnaly we create our first generic chain
chain_chat = LLMChain(llm=llm, prompt=prompt)
chain_chat({"question": QUESTION, "language": "English"})

# Text-based Indexes that we are going to query (from Notebook 01 and 02)
index1_name = "abc-index-files"
index2_name = "abcd-index-csv"
indexes = [index1_name,index2_name  ]


k = 10 # Number of results per each text_index
ordered_results = get_search_results(QUESTION, indexes, k=10, reranker_threshold=1)
print("Number of results:",len(ordered_results))

embedder = OpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1) 

#%%time
for key,value in ordered_results.items():
    if value["vectorized"] != True: # If the document has not been vectorized yet
        i = 0
        print("Vectorizing",len(value["chunks"]),"chunks from Document:",value["location"])
        for chunk in value["chunks"]: # Iterate over the document's text chunks
            try:
                upload_payload = {  # Insert the chunk and its vector in the vector-based index
                    "value": [
                        {
                            "id": key + "_" + str(i),
                            "title": f"{value['title']}_chunk_{str(i)}",
                            "chunk": chunk,
                            "chunkVector": embedder.embed_query(chunk if chunk!="" else "-------"),
                            "name": value["name"],
                            "location": value["location"],
                            "@search.action": "upload"
                        },
                    ]
                }

                r = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + value["index"]+"-vector" + "/docs/index",
                                     data=json.dumps(upload_payload), headers=headers, params=params)
                
                if r.status_code != 200:
                    print(r.status_code)
                    print(r.text)
                else:
                    i = i + 1 # increment chunk number
                    
                    # Update document in text-based index and mark it as "vectorized"
                    upload_payload = {
                        "value": [
                            {
                                "id": key,
                                "vectorized": True,
                                "@search.action": "merge"
                            },
                        ]
                    }

                    r = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + value["index"]+ "/docs/index",
                                     data=json.dumps(upload_payload), headers=headers, params=params)
                    
                    
            except Exception as e:
                print("Exception:",e)
                print(content)
                continue
            

vector_indexes = [index+"-vector" for index in indexes]

k = 10
similarity_k = 3
ordered_results = get_search_results(QUESTION, vector_indexes,
                                        k=k, # Number of results per vector index
                                        reranker_threshold=1,
                                        vector_search=True, 
                                        similarity_k=similarity_k,
                                        query_vector = embedder.embed_query(QUESTION)
                                        )
print("Number of results:",len(ordered_results))

top_docs = []
for key,value in ordered_results.items():
    location = value["location"] if value["location"] is not None else ""
    top_docs.append(Document(page_content=value["chunk"], metadata={"source": location}))
        
print("Number of chunks:",len(top_docs))

# Calculate number of tokens of our docs
if(len(top_docs)>0):
    tokens_limit = model_tokens_limit(MODEL) # this is a custom function we created in common/utils.py
    prompt_tokens = num_tokens_from_string(COMBINE_PROMPT_TEMPLATE) # this is a custom function we created in common/utils.py
    context_tokens = num_tokens_from_docs(top_docs) # this is a custom function we created in common/utils.py
    
    requested_tokens = prompt_tokens + context_tokens + COMPLETION_TOKENS
    
    chain_type = "map_reduce" if requested_tokens > 0.9 * tokens_limit else "stuff"  
    
    print("System prompt token count:",prompt_tokens)
    print("Max Completion Token count:", COMPLETION_TOKENS)
    print("Combined docs (context) token count:",context_tokens)
    print("--------")
    print("Requested token count:",requested_tokens)
    print("Token limit for", MODEL, ":", tokens_limit)
    print("Chain Type selected:", chain_type)
        
else:
    print("NO RESULTS FROM AZURE SEARCH")
    
if chain_type == "stuff":
    chain = load_qa_with_sources_chain(llm, chain_type=chain_type, 
                                       prompt=COMBINE_PROMPT)
elif chain_type == "map_reduce":
    chain = load_qa_with_sources_chain(llm, chain_type=chain_type, 
                                       question_prompt=COMBINE_QUESTION_PROMPT,
                                       combine_prompt=COMBINE_PROMPT,
                                       return_intermediate_steps=True)
    
#%%time
# Try with other language as well
response = chain({"input_documents": top_docs, "question": QUESTION, "language": "English"})

display(Markdown(response['output_text']))

