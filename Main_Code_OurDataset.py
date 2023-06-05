#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[2]:


# !pip install git+https://github.com/deepset-ai/haystack.git


# In[3]:


# !pip install spacy


# In[2]:


from haystack.utils import clean_wiki_text, fetch_archive_from_http, print_answers, print_documents
from haystack.nodes import FARMReader, TransformersReader


# In[3]:


import torch
torch.cuda.empty_cache()


# In[4]:


import os
import spacy 

SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
nlp = spacy.load(SPACY_MODEL, disable = ["ner","parser","textcat"])

print("loaded spacy model")


# In[5]:


import nltk
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

def processSentence(sentence):
    global nlp
    #Lowering sentences
    sentence = sentence.lower() 

    #Removing punctuations
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    sentence = sentence.translate(str.maketrans(symbols,' '*len(symbols)))

    #removing stopwords
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(sentence)  
    filtered_sentence = " ".join([w for w in word_tokens if len(w) > 1])

    #lemmatization
    doc = nlp(filtered_sentence)
    lemmatized_sentence = [t.lemma_ for t in doc]
    
    return lemmatized_sentence


# In[6]:


#fomatting data stored in json 
import json
from pprint import pprint
data = json.load(open('./data/our_dataset.json', 'r'))
inputDicts = []
for keys in data:
    p = data[keys]['paragraphs']
    preprocessP = processSentence(p)
    processP = ' '.join(preprocessP)

    if processP != '' and len(processP.split()) > 10:
        inputDicts.append({'content': processP, 'meta': {'name': keys, 'cleanContent': p}})

    
print("DONE INPUTDICTS")

# In[7]:


from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TfidfRetriever

document_store = InMemoryDocumentStore(embedding_dim=768)
document_store.write_documents(inputDicts)

print("created document store")


# In[8]:


def model_name(full_name):
    idx = full_name.find('/')
    model_name = full_name[idx+1:]
    return model_name


# In[9]:


emd_model='flax-sentence-embeddings/all_datasets_v3_mpnet-base'


# In[10]:


from haystack.nodes import EmbeddingRetriever
retriever = EmbeddingRetriever(
   document_store=document_store,
   embedding_model=emd_model,
   model_format="sentence_transformers",
)


# In[11]:


document_store.update_embeddings(retriever,batch_size=128)
print("updated embeddings")


# In[12]:


from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents


# In[13]:


gen_model='vblagoje/bart_lfqa'


# In[14]:


from haystack.nodes import Seq2SeqGenerator

generator = Seq2SeqGenerator(model_name_or_path=gen_model,use_gpu=True)


# In[15]:


from haystack.pipelines import GenerativeQAPipeline

pipe = GenerativeQAPipeline(generator = generator, retriever = retriever)

print("created pipeline")
# In[16]:


import pandas as pd
eli5_data= pd.read_csv('./data/Our_Dataset.csv')
Questions=eli5_data['questions']


# In[17]:


answers = []


# In[ ]:


for question in Questions:
    res = pipe.run(query=question, params={"Generator": {"top_k":1}, "Retriever": {"top_k": 10}})
    answers.append(res['answers'][0].answer)

print("done with answers")


# In[ ]:


import pandas as pd
prediction = pd.DataFrame(columns=['Query', 'Answer_True','Our_Ans'])
prediction['Query'] = Questions
prediction['Answer_True']=eli5_data['answers']
prediction['Our_Ans'] = answers
prediction.to_csv('./Our_dataset_material/our_dataset_our_model_final.csv')

print("Saved")


# In[ ]:


# true_data = pd.read_csv("./data/Hopon_True.csv")

# for index, row in true_data.iterrows():
#     p = row['Answer']
#     preprocessP = processSentence(p)
#     processP = ' '.join(preprocessP)
#     true_data.loc[index, 'Answer'] = processP
    
# true_data


# In[ ]:


# predicted_ans = prediction['Answer']
# true_ans = true_data['Answer']


# In[ ]:


# def compare(prediction, truth):
#     pred_tokens = prediction.split()
#     truth_tokens = truth.split()
    
#     if len(pred_tokens) == 0 or len(truth_tokens) == 0:
#         print(int(pred_tokens == truth_tokens))
#         return int(pred_tokens == truth_tokens)
    
#     common_tokens = set(pred_tokens) & set(truth_tokens)
    
#     # if there are no common tokens then f1 = 0
#     if len(common_tokens) == 0:
#         print('Precision score:', 0)
#         print('Recall score:', 0)
#         print('F1 Score:', 0)
#         return 0
    
    
#     prec = len(common_tokens) / len(pred_tokens)
#     rec = len(common_tokens) / len(truth_tokens)
#     f1_score = 2 * (prec * rec) / (prec + rec)
#     print('Precision score:', prec)
#     print('Recall score:', rec)
#     print('F1 Score:', f1_score)
#     return 0
    

# def compute_exact_match(prediction, truth):
#     return int(prediction == truth)


# In[ ]:


# print('Comparision scores - ')
# for i in range(len(predicted_ans)):
#     p = predicted_ans[i]
#     a = true_ans[i]
# #     em_score = compute_exact_match(p, a)
# #     f1_score = compute_f1(p, a)
    
# #     print(em_score)
# #     print('Query :', Questions[i])
# #     print('Predicted answer :', p)
#     print('Q', i+1, '-')
#     compare(p, a)


# In[ ]:




