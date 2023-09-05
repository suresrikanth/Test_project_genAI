## Set up variables
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

##Multi-Index Search queries

# Text-based Indexes that we are going to query (from Notebook 01 and 02)
index1_name = "kaegp-index-files"
index2_name = "kaega-index-csv"
indexes = [index1_name,index2_name ]

QUESTION = "What is CLP?"

## Search on both indexes individually and aggragate results
''' In order to standarize the indexes, there must be 8 mandatory fields
present on each text-based index: id, title, content, chunks, language, 
name, location, vectorized. This is so that each document can be treated 
the same along the code. Also, all indexes must have a semantic configuration.'''

agg_search_results = dict()

for index in indexes:
    search_payload = {
        "search": QUESTION,
        "select": "id, title, language",
        "queryType": "semantic",
        "semanticConfiguration": "my-semantic-config",
        "count": "true",
        "speller": "lexicon",
        "queryLanguage": "en-us",
        "captions": "extractive",
        "answers": "extractive",
        "top": "2"
    }

    r = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search",
                     data=json.dumps(search_payload), headers=headers, params=params)
    print(r.status_code)

    search_results = r.json()
    agg_search_results[index]=search_results
    print("Index:", index, "Results Found: {}, Results Returned: {}".format(search_results['@odata.count'], len(search_results['value'])))


## Display the top results (from both searches) based on the score

display(HTML('<h4>Top Answers</h4>'))

for index,search_results in agg_search_results.items():
    for result in search_results['@search.answers']:
        if result['score'] > 0.5: # Show answers that are at least 50% of the max possible score=1
            display(HTML('<h5>' + 'Answer - score: ' + str(round(result['score'],2)) + '</h5>'))
            display(HTML(result['text']))
            
print("\n\n")
display(HTML('<h4>Top Results</h4>'))

content = dict()
ordered_content = OrderedDict()


for index,search_results in agg_search_results.items():
    for result in search_results['value']:
        if result['@search.rerankerScore'] > 1:# Show answers that are at least 25% of the max possible score=4
            content[result['id']]={
                                    "title": result['title'], 
                                    "caption": result['@search.captions'][0]['text'],
                                    "score": result['@search.rerankerScore'],
                                    "index": index
                                    }
    
#After results have been filtered we will Sort and add them as an Ordered list\n",
for id in sorted(content, key= lambda x: content[x]["score"], reverse=True):
    ordered_content[id] = content[id]
    url = os.environ['BLOB_SAS_TOKEN']
    title = str(ordered_content[id]['title']) if (ordered_content[id]['title']) else "Answer"
    score = str(round(ordered_content[id]['score'],2))
    display(HTML('<h5><a href="'+ url + '">' + title + '</a> - score: '+ score + '</h5>'))
    display(HTML(ordered_content[id]['caption']))
