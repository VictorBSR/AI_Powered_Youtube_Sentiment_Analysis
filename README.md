# AI-Powered YouTube Sentiment Scanner <a class="anchor" id="top-bullet"></a>

Victor B. S. Reis

Sep, 2024

## The Problem
- Small business owners, startups, and marketing teams are eager to understand how their products or services are being received in the market
- They wish for insights into customer opinions and trends, particularly on platforms like YouTube, to gauge performance and evaluate marketing KPIs
- Whether it's tracking product popularity or identifying sentiment, these businesses rely on real-time feedback to make data-driven decisions!

## Project Overview
This project aims to help users discover and analyze customer sentiment from Youtube video comments by:
- Searching for a given term, for example a product name
- Presenting the resulting video list from the research, while labeling each title as relevant or not
- Each video will have their comments analyzed based on their content
- Using NLP and AI, each comments will be classified under specific sentiment categories (Positive, Negative or Neutral)
- Presenting the sentiment analysis results via word clouds and brief descriptions that highlight key insights from the data

## Steps
- **Search for an expression/name (YouTube API):**
    - The YouTube Data API is used to retrieve a list of videos related to the search term
    - Python’s requests library interact with the API and parse the video metadata for further analysis
    
- **Classify Relevant Videos (ChatGPT/LLM Classification):**
    - Each video returned from the search is filtered by relevance using an LLM
    - Python’s openai library interacts with the LLM, where the video title and description are sent to classify whether the video is relevant or not (using a custom prompt).
    
- **Pre-process Comments:**
    - Once relevant videos are selected, the project uses the YouTube API to retrieve comments from each video.
    - Text preprocessing steps like tokenization, removal of stop words, and basic cleaning are performed using libraries like nltk or spacy.

- **Classify Sentiment (ChatGPT/LLM Sentiment Analysis):**
    - Comments from each video are passed to the LLM, which classifies the overall sentiment of each one as Positive, Neutral, or Negative.
    - This is accomplished by using the OpenAI library to communicate with the LLM, with a customizable prompt to reflect the video’s context.
    - Results are saved into a Pandas DataFrame for structured analysis.
    
- **Word Cloud Generation:**
    - The most frequent terms in positive and negative comments are visualized in a word cloud.
    - Python’s wordcloud library is used to create these visualizations, offering a quick overview of the most discussed terms.
    
- **Final Summary and Insights (Sentiment & Summary Generation):**
    - For each video, a brief summary is generated based on the most frequent sentiments and key opinions.
    - The LLM produces a concise description highlighting significant trends (e.g., “Most users love the product’s design but complain about its price”).
    - The overall sentiment rating is aggregated and visualized for user interpretation.


## Table of Contents:
* [Preparing Inputs](#second-bullet)
* [Pre-processing Comments](#third-bullet)
* [Classifying Sentiment](#fourth-bullet)
* [Deployment in Production](#fifth-bullet)
* [Results and Future Improvements](#sixth-bullet)

---

# Preparing Inputs<a class="anchor" id="second-bullet"></a>
The first step is to search for video results and classifying their relevance based on a search expression, like a product name.

### Imports


```python
import pandas as pd
import numpy as np
import nltk
import emoji
import re
import requests
import os
from openai import OpenAI
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from langdetect import detect, DetectorFactory
from googleapiclient.discovery import build
```


```python
# keys 
api_key = ''
openai_key = ''
```

### Loading API and testing retrieval
First we try to initialize the youtube API based on the official documentation and return all the comments from a video based on its ID


```python
def load_api(api_key):
    # Set up the API
    youtube = build('youtube', 'v3', developerKey=api_key)
    return youtube

# Fetch comments
youtube = load_api(api_key)
```


```python
# Example usage
video_id = 'i1GXJFH8xm4'

# make a request for the comment threads
comments = []
comments_request = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    maxResults=100
)
response = comments_request.execute()

# make a request for video info
video_request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
video_response = video_request.execute()
```


```python
video_response
```




    {'kind': 'youtube#videoListResponse',
     'etag': '0vp-SAtkppYc4kFLUa634H0a08E',
     'items': [{'kind': 'youtube#video',
       'etag': 'm5A9pAdnQQb7MzByQNXs9dr05D4',
       'id': 'i1GXJFH8xm4',
       'snippet': {'publishedAt': '2024-09-03T14:45:04Z',
        'channelId': 'UCtaDOcil7AXjT28K0mW7UQA',
        'title': 'COMO EXIBIR IMAGENS DA COLUNA DE IMAGEM DO SHAREPOINT LIST ONLINE NO POWER BI PASSO A PASSO',
        'description': 'Aprenda passo a passo como exibir imagens da coluna de imagem do sharepoint list online direto no power bi.\n\nConheça nossos cursos e nos apoie!\nhttps://ead.vicotreinamentos.com.br\n\n00:00 Introdução , conectando sharepoint list ao power bi passo a passo v1 e v2\n02:46 Como chegar na URL da imagem da coluna da lista.\n04:35 Criando coluna calculada com a URL para obter imagem da coluna\n06:35 Criando função para obter dados da URL.\n08:50 Parte mais importante .\n\n--\n\n📚 GRUPO DE ESTUDOS E DUVIDAS GRATUITO\n- Telegram: https://vicotreinamentos.com.br/GrupoEstudoGratis\n\n🔽 SE CONECTE COMIGO\n- Instagram:   https://www.instagram.com/ronanvico/ \n- LinkedIn: https://www.linkedin.com/in/ronan-vico/\n- Canal Telegram (Materiais): https://t.me/RonanVico\n\n🎁 FERRAMENTAS E DESCONTOS\n✅ 📚 Descontos no meus cursos | https://vicotreinamentos.com.br/QueroDesconto\n\n🙌 AJUDE O CANAL\n- Aperte o botão VALEU nos meus Vídeos.\n- Vire membro do canal, nos pague um café☕ por mês!\n- Invista nos cursos ou compartilhe o canal!\n\n🙏 SOLICITE UM VÍDEO PARA O CANAL \n- https://vicotreinamentos.com.br/TemaDeVideo\n\n🎲 CONSULTORIA / CONTATO EMPRESARIAL \n- contato@vicotreinamentos.com.br',
        'thumbnails': {'default': {'url': 'https://i.ytimg.com/vi/i1GXJFH8xm4/default.jpg',
          'width': 120,
          'height': 90},
         'medium': {'url': 'https://i.ytimg.com/vi/i1GXJFH8xm4/mqdefault.jpg',
          'width': 320,
          'height': 180},
         'high': {'url': 'https://i.ytimg.com/vi/i1GXJFH8xm4/hqdefault.jpg',
          'width': 480,
          'height': 360},
         'standard': {'url': 'https://i.ytimg.com/vi/i1GXJFH8xm4/sddefault.jpg',
          'width': 640,
          'height': 480},
         'maxres': {'url': 'https://i.ytimg.com/vi/i1GXJFH8xm4/maxresdefault.jpg',
          'width': 1280,
          'height': 720}},
        'channelTitle': 'Ronan Vico',
        'tags': ['analytics',
         'microsoft',
         'big data',
         'data analytics',
         'power bi',
         'powerbi',
         'sharepoint',
         'imagens do sharepoint',
         'imagens do sharepoint no power bi',
         'como exibir imagens do sharepoint no power bi',
         'dashboard',
         'dashboards com imagens do sharepoint',
         'dashboards com imagens',
         'dashboard com imagem',
         'como exibir imagens do sharepoint em um dashboard do power bi',
         'power bi com sharepoint',
         'integrar sharepoint no power bi',
         'sharepoint power bi',
         'COLUNA IMAGEM',
         'coluna de imagem',
         'coluna de imagem no power bi',
         'lists'],
        'categoryId': '28',
        'liveBroadcastContent': 'none',
        'defaultLanguage': 'pt-BR',
        'localized': {'title': 'COMO EXIBIR IMAGENS DA COLUNA DE IMAGEM DO SHAREPOINT LIST ONLINE NO POWER BI PASSO A PASSO',
         'description': 'Aprenda passo a passo como exibir imagens da coluna de imagem do sharepoint list online direto no power bi.\n\nConheça nossos cursos e nos apoie!\nhttps://ead.vicotreinamentos.com.br\n\n00:00 Introdução , conectando sharepoint list ao power bi passo a passo v1 e v2\n02:46 Como chegar na URL da imagem da coluna da lista.\n04:35 Criando coluna calculada com a URL para obter imagem da coluna\n06:35 Criando função para obter dados da URL.\n08:50 Parte mais importante .\n\n--\n\n📚 GRUPO DE ESTUDOS E DUVIDAS GRATUITO\n- Telegram: https://vicotreinamentos.com.br/GrupoEstudoGratis\n\n🔽 SE CONECTE COMIGO\n- Instagram:   https://www.instagram.com/ronanvico/ \n- LinkedIn: https://www.linkedin.com/in/ronan-vico/\n- Canal Telegram (Materiais): https://t.me/RonanVico\n\n🎁 FERRAMENTAS E DESCONTOS\n✅ 📚 Descontos no meus cursos | https://vicotreinamentos.com.br/QueroDesconto\n\n🙌 AJUDE O CANAL\n- Aperte o botão VALEU nos meus Vídeos.\n- Vire membro do canal, nos pague um café☕ por mês!\n- Invista nos cursos ou compartilhe o canal!\n\n🙏 SOLICITE UM VÍDEO PARA O CANAL \n- https://vicotreinamentos.com.br/TemaDeVideo\n\n🎲 CONSULTORIA / CONTATO EMPRESARIAL \n- contato@vicotreinamentos.com.br'},
        'defaultAudioLanguage': 'pt-BR'}}],
     'pageInfo': {'totalResults': 1, 'resultsPerPage': 1}}



### Search for video results

After obtaining a good understanding on how the data from a video is returned from the API, now we create a function to search for multiple videos based on a search expression:


```python
def search_videos(search_query, api_key):
    max_results = 10  # Number of videos to retrieve
    API_KEY = api_key

    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults={max_results}&q={search_query}&type=video&key={API_KEY}"

    response = requests.get(url)
    data = response.json()

    # Extract video IDs and URLs
    video_data = []
    for item in data['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        video_description = item['snippet']['description']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # append to the list
        video_data.append({
            'video_id': video_id,
            'video_title': video_title,
            'video_description': video_description,
            'video_url': video_url
        })
        
    return pd.DataFrame(video_data)
```


```python
# quick test
df_videos = search_videos("Ray-Ban Meta Smart Glasses", api_key)
display(df_videos)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>video_title</th>
      <th>video_description</th>
      <th>video_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E1LW_MteTho</td>
      <td>Introducing the Ray-Ban Meta Smart Glasses Col...</td>
      <td>Meet the next generation of smart glasses. The...</td>
      <td>https://www.youtube.com/watch?v=E1LW_MteTho</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ViA4-YWx8Y4</td>
      <td>Meta Ray-Ban Smart Glasses Review - 6 Months L...</td>
      <td>Have the Meta Ray-Ban Smart Glasses been worth...</td>
      <td>https://www.youtube.com/watch?v=ViA4-YWx8Y4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pgiWqkvIclk</td>
      <td>The Ray-Ban Meta Smart Glasses are a turning p...</td>
      <td>Smart glasses have always been a hard sell, bu...</td>
      <td>https://www.youtube.com/watch?v=pgiWqkvIclk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>utSc7DxIvTU</td>
      <td>RayBan Meta SMART GLASSES Review and MY HUGE M...</td>
      <td>I made a HUGE mistake when ordering the RayBan...</td>
      <td>https://www.youtube.com/watch?v=utSc7DxIvTU</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gY9cP2ZWfVI</td>
      <td>Ray-Ban Meta smart glasses hands-on: Techy sun...</td>
      <td>Meta has almost completely revamped its high-t...</td>
      <td>https://www.youtube.com/watch?v=gY9cP2ZWfVI</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ykj-ZQDrgYA</td>
      <td>What Ray-Ban Meta Smart Glasses Are ACTUALLY Like</td>
      <td>Ray-Ban Meta https://geni.us/jZMqUE The mic I ...</td>
      <td>https://www.youtube.com/watch?v=ykj-ZQDrgYA</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ct6f28FL8s8</td>
      <td>COOLEST AI smart glasses are ever🤯#raybanmeta</td>
      <td>This is Called the meta rayban glasses that ar...</td>
      <td>https://www.youtube.com/watch?v=ct6f28FL8s8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>yj9_lRvbbNU</td>
      <td>Rayban Meta Smart Glasses cannot secretly reco...</td>
      <td></td>
      <td>https://www.youtube.com/watch?v=yj9_lRvbbNU</td>
    </tr>
    <tr>
      <th>8</th>
      <td>iivBLI8ml6o</td>
      <td>Anderson .Paak, Tinashe, and James Blake exper...</td>
      <td>AndersonPaak, @tinashenow, and @jamesblake are...</td>
      <td>https://www.youtube.com/watch?v=iivBLI8ml6o</td>
    </tr>
    <tr>
      <th>9</th>
      <td>_PhGQLIztog</td>
      <td>Why I RETURNED the Ray Ban Meta Smart Glasses ...</td>
      <td>The Ray Ban Meta smart glasses have a tiny cam...</td>
      <td>https://www.youtube.com/watch?v=_PhGQLIztog</td>
    </tr>
  </tbody>
</table>
</div>


### Evaluating relevance with AI

With the video list, maybe not all of them will contain relevant info or relevant comments for our purpose. We need a way to quickly determine which videos will be relevant or not to the analysis. For this, a LLM will be used, and for a matter of simplicity and cost at this moment OpenAI's API is a good choice, and GPT 3.5 should suffice.

This was thought more as an optional step in order to allow the user double check which videos are about to be analysed and make sure none of them are unrelated to the desired topic, avoiding contamination of the upcoming sentiment analysis.


```python
# function to classify video title as relevant or not for our search
def classify_video(search_query, df_videos, api_key):
    # prompt
    prompt_sistema = f"""You are a marketing analyst skilled in evaluating the relevance of videos for product or topic analysis.
    Given a video title and description, determine if the content is RELEVANT or NOT RELEVANT based on its value for marketing 
    insights, research purposes, or general audience interest. Consider aspects such as alignment with the search term or 
    expression, potential to engage or inform the target audience, and contribution to a comprehensive understanding of the 
    topic. Your goal is to help filtering those videos so stakeholders can understand how their products or services are being 
    received in the market. You will receive a "search_string" for a product or topic that will be researched, as well as a 
    video title and its description. You should analyze each one of them and return only a single string with one of the options:
    ["RELEVANT","NOT RELEVANT"]
    """
    # initialize client
    cliente = OpenAI(api_key=api_key)
    
    # iterate every dataframe row
    for _,item in df_videos.iterrows():
        prompt_user = f"Search query: {search_query}\nVideo title: {item['video_title']}\nVideo description: {item['video_description']}"

        response = cliente.chat.completions.create(
            messages = [
                {
                    "role":"system",
                    "content":prompt_sistema
                },
                {
                    "role":"user",
                    "content":prompt_user
                }
            ],
            model="gpt-3.5-turbo",
            max_tokens=10
        )

        result = response.choices[0].message.content
        df_videos['relevance'] = result
        
        
    return df_videos
```


```python
# quick test
classify_video("Ray-Ban Meta Smart Glasses", df_videos, openai_key)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>video_title</th>
      <th>video_description</th>
      <th>video_url</th>
      <th>relevance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E1LW_MteTho</td>
      <td>Introducing the Ray-Ban Meta Smart Glasses Col...</td>
      <td>Meet the next generation of smart glasses. The...</td>
      <td>https://www.youtube.com/watch?v=E1LW_MteTho</td>
      <td>NOT RELEVANT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ViA4-YWx8Y4</td>
      <td>Meta Ray-Ban Smart Glasses Review - 6 Months L...</td>
      <td>Have the Meta Ray-Ban Smart Glasses been worth...</td>
      <td>https://www.youtube.com/watch?v=ViA4-YWx8Y4</td>
      <td>NOT RELEVANT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pgiWqkvIclk</td>
      <td>The Ray-Ban Meta Smart Glasses are a turning p...</td>
      <td>Smart glasses have always been a hard sell, bu...</td>
      <td>https://www.youtube.com/watch?v=pgiWqkvIclk</td>
      <td>NOT RELEVANT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>utSc7DxIvTU</td>
      <td>RayBan Meta SMART GLASSES Review and MY HUGE M...</td>
      <td>I made a HUGE mistake when ordering the RayBan...</td>
      <td>https://www.youtube.com/watch?v=utSc7DxIvTU</td>
      <td>NOT RELEVANT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gY9cP2ZWfVI</td>
      <td>Ray-Ban Meta smart glasses hands-on: Techy sun...</td>
      <td>Meta has almost completely revamped its high-t...</td>
      <td>https://www.youtube.com/watch?v=gY9cP2ZWfVI</td>
      <td>NOT RELEVANT</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ykj-ZQDrgYA</td>
      <td>What Ray-Ban Meta Smart Glasses Are ACTUALLY Like</td>
      <td>Ray-Ban Meta https://geni.us/jZMqUE The mic I ...</td>
      <td>https://www.youtube.com/watch?v=ykj-ZQDrgYA</td>
      <td>NOT RELEVANT</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ct6f28FL8s8</td>
      <td>COOLEST AI smart glasses are ever🤯#raybanmeta</td>
      <td>This is Called the meta rayban glasses that ar...</td>
      <td>https://www.youtube.com/watch?v=ct6f28FL8s8</td>
      <td>NOT RELEVANT</td>
    </tr>
    <tr>
      <th>7</th>
      <td>yj9_lRvbbNU</td>
      <td>Rayban Meta Smart Glasses cannot secretly reco...</td>
      <td></td>
      <td>https://www.youtube.com/watch?v=yj9_lRvbbNU</td>
      <td>NOT RELEVANT</td>
    </tr>
    <tr>
      <th>8</th>
      <td>iivBLI8ml6o</td>
      <td>Anderson .Paak, Tinashe, and James Blake exper...</td>
      <td>AndersonPaak, @tinashenow, and @jamesblake are...</td>
      <td>https://www.youtube.com/watch?v=iivBLI8ml6o</td>
      <td>NOT RELEVANT</td>
    </tr>
    <tr>
      <th>9</th>
      <td>_PhGQLIztog</td>
      <td>Why I RETURNED the Ray Ban Meta Smart Glasses ...</td>
      <td>The Ray Ban Meta smart glasses have a tiny cam...</td>
      <td>https://www.youtube.com/watch?v=_PhGQLIztog</td>
      <td>NOT RELEVANT</td>
    </tr>
  </tbody>
</table>
</div>



### Adding more videos

We also want the user to add new video URLs aside from those if they want to, so we make another function for that:


```python
def add_videos(video_url, df_videos, api_key):
    API_KEY = api_key
    
     # extract the video ID from the URL
    video_id = video_url.split('v=')[-1] 
    
    url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=snippet&key={API_KEY}"

    response = requests.get(url)
    data = response.json()
    
    # Extract the necessary details
    video_title = data['items'][0]['snippet']['title']
    video_description = data['items'][0]['snippet']['description']
    
    # Append the new video information to the DataFrame
    new_row = {
        'video_id': video_id,
        'video_title': video_title,
        'video_description': video_description,
        'video_url': video_url
    }
    new_df = pd.DataFrame([new_row])
    df_videos = pd.concat([df_videos, new_df], axis=0, ignore_index=True)

    return df_videos
```


```python
# Testing everything together

# search for a product
search_query = "nike air review"
df_videos = search_videos(search_query, api_key)
display(df_videos)

# evaluate relevance
df_videos = classify_video(search_query, df_videos, openai_key)
display(df_videos)

# add new URLs:
df_videos = add_videos("https://www.youtube.com/watch?v=hb0j9Qn-KjM", df_videos, api_key)
df_videos = add_videos("https://www.youtube.com/watch?v=M-XfdqPBAMU", df_videos, api_key)
display(df_videos)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>video_title</th>
      <th>video_description</th>
      <th>video_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8y5-UvxoD2E</td>
      <td>You’re wrong about Nike Air Max 270</td>
      <td>If you're ever injured in an accident, you can...</td>
      <td>https://www.youtube.com/watch?v=8y5-UvxoD2E</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5JzIrzXV2vI</td>
      <td>Pros &amp;amp; Cons: 2024 Nike Air Max DN Review!</td>
      <td>Shop Hibbett City Gear Here! https://bit.ly/37...</td>
      <td>https://www.youtube.com/watch?v=5JzIrzXV2vI</td>
    </tr>
    <tr>
      <th>2</th>
      <td>p_-2u8x4Re0</td>
      <td>Nike Alphafly Next % 2 Shoe Review</td>
      <td></td>
      <td>https://www.youtube.com/watch?v=p_-2u8x4Re0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P2YQHLY8TGw</td>
      <td>Nike Air Max 270 Review Black and White</td>
      <td>The Nike Air Max 270 was originally released i...</td>
      <td>https://www.youtube.com/watch?v=P2YQHLY8TGw</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xGM6g7HTInU</td>
      <td>Nike Air More Uptempo Low</td>
      <td>Shop POIZON here!! #ad #poizon Use my code [TE...</td>
      <td>https://www.youtube.com/watch?v=xGM6g7HTInU</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xxrXtdVyn34</td>
      <td>Nike Doesn&amp;#39;t Know What&amp;#39;s Inside Their ...</td>
      <td>Buy some Rose anvil leather goods that EVERYBO...</td>
      <td>https://www.youtube.com/watch?v=xxrXtdVyn34</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4iv34wHa8Jk</td>
      <td>Nike Air Max 97 White Review: Not What I Expec...</td>
      <td>Description In this video, we review the Air M...</td>
      <td>https://www.youtube.com/watch?v=4iv34wHa8Jk</td>
    </tr>
    <tr>
      <th>7</th>
      <td>K-HoJUGc4XM</td>
      <td>Nike Air Deldon Biggest Pros And Cons ( Perfor...</td>
      <td>Grab a pair at Nike: https://geni.us/deldon FR...</td>
      <td>https://www.youtube.com/watch?v=K-HoJUGc4XM</td>
    </tr>
    <tr>
      <th>8</th>
      <td>X1ayLlbBtrg</td>
      <td>Nike Jordan 3 Retro Cement Grey | Unboxing &amp;am...</td>
      <td></td>
      <td>https://www.youtube.com/watch?v=X1ayLlbBtrg</td>
    </tr>
    <tr>
      <th>9</th>
      <td>wKEF7-LNhSo</td>
      <td>BEST AIR MAX UNDER £100!? Nike Air Max &amp;quot;S...</td>
      <td>Nike Air Max SYSTM DM9537-001 https://tidd.ly/...</td>
      <td>https://www.youtube.com/watch?v=wKEF7-LNhSo</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>video_title</th>
      <th>video_description</th>
      <th>video_url</th>
      <th>relevance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8y5-UvxoD2E</td>
      <td>You’re wrong about Nike Air Max 270</td>
      <td>If you're ever injured in an accident, you can...</td>
      <td>https://www.youtube.com/watch?v=8y5-UvxoD2E</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5JzIrzXV2vI</td>
      <td>Pros &amp;amp; Cons: 2024 Nike Air Max DN Review!</td>
      <td>Shop Hibbett City Gear Here! https://bit.ly/37...</td>
      <td>https://www.youtube.com/watch?v=5JzIrzXV2vI</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>p_-2u8x4Re0</td>
      <td>Nike Alphafly Next % 2 Shoe Review</td>
      <td></td>
      <td>https://www.youtube.com/watch?v=p_-2u8x4Re0</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P2YQHLY8TGw</td>
      <td>Nike Air Max 270 Review Black and White</td>
      <td>The Nike Air Max 270 was originally released i...</td>
      <td>https://www.youtube.com/watch?v=P2YQHLY8TGw</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xGM6g7HTInU</td>
      <td>Nike Air More Uptempo Low</td>
      <td>Shop POIZON here!! #ad #poizon Use my code [TE...</td>
      <td>https://www.youtube.com/watch?v=xGM6g7HTInU</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xxrXtdVyn34</td>
      <td>Nike Doesn&amp;#39;t Know What&amp;#39;s Inside Their ...</td>
      <td>Buy some Rose anvil leather goods that EVERYBO...</td>
      <td>https://www.youtube.com/watch?v=xxrXtdVyn34</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4iv34wHa8Jk</td>
      <td>Nike Air Max 97 White Review: Not What I Expec...</td>
      <td>Description In this video, we review the Air M...</td>
      <td>https://www.youtube.com/watch?v=4iv34wHa8Jk</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>7</th>
      <td>K-HoJUGc4XM</td>
      <td>Nike Air Deldon Biggest Pros And Cons ( Perfor...</td>
      <td>Grab a pair at Nike: https://geni.us/deldon FR...</td>
      <td>https://www.youtube.com/watch?v=K-HoJUGc4XM</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>8</th>
      <td>X1ayLlbBtrg</td>
      <td>Nike Jordan 3 Retro Cement Grey | Unboxing &amp;am...</td>
      <td></td>
      <td>https://www.youtube.com/watch?v=X1ayLlbBtrg</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>9</th>
      <td>wKEF7-LNhSo</td>
      <td>BEST AIR MAX UNDER £100!? Nike Air Max &amp;quot;S...</td>
      <td>Nike Air Max SYSTM DM9537-001 https://tidd.ly/...</td>
      <td>https://www.youtube.com/watch?v=wKEF7-LNhSo</td>
      <td>RELEVANT</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>video_title</th>
      <th>video_description</th>
      <th>video_url</th>
      <th>relevance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8y5-UvxoD2E</td>
      <td>You’re wrong about Nike Air Max 270</td>
      <td>If you're ever injured in an accident, you can...</td>
      <td>https://www.youtube.com/watch?v=8y5-UvxoD2E</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5JzIrzXV2vI</td>
      <td>Pros &amp;amp; Cons: 2024 Nike Air Max DN Review!</td>
      <td>Shop Hibbett City Gear Here! https://bit.ly/37...</td>
      <td>https://www.youtube.com/watch?v=5JzIrzXV2vI</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>p_-2u8x4Re0</td>
      <td>Nike Alphafly Next % 2 Shoe Review</td>
      <td></td>
      <td>https://www.youtube.com/watch?v=p_-2u8x4Re0</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P2YQHLY8TGw</td>
      <td>Nike Air Max 270 Review Black and White</td>
      <td>The Nike Air Max 270 was originally released i...</td>
      <td>https://www.youtube.com/watch?v=P2YQHLY8TGw</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xGM6g7HTInU</td>
      <td>Nike Air More Uptempo Low</td>
      <td>Shop POIZON here!! #ad #poizon Use my code [TE...</td>
      <td>https://www.youtube.com/watch?v=xGM6g7HTInU</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xxrXtdVyn34</td>
      <td>Nike Doesn&amp;#39;t Know What&amp;#39;s Inside Their ...</td>
      <td>Buy some Rose anvil leather goods that EVERYBO...</td>
      <td>https://www.youtube.com/watch?v=xxrXtdVyn34</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4iv34wHa8Jk</td>
      <td>Nike Air Max 97 White Review: Not What I Expec...</td>
      <td>Description In this video, we review the Air M...</td>
      <td>https://www.youtube.com/watch?v=4iv34wHa8Jk</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>7</th>
      <td>K-HoJUGc4XM</td>
      <td>Nike Air Deldon Biggest Pros And Cons ( Perfor...</td>
      <td>Grab a pair at Nike: https://geni.us/deldon FR...</td>
      <td>https://www.youtube.com/watch?v=K-HoJUGc4XM</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>8</th>
      <td>X1ayLlbBtrg</td>
      <td>Nike Jordan 3 Retro Cement Grey | Unboxing &amp;am...</td>
      <td></td>
      <td>https://www.youtube.com/watch?v=X1ayLlbBtrg</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>9</th>
      <td>wKEF7-LNhSo</td>
      <td>BEST AIR MAX UNDER £100!? Nike Air Max &amp;quot;S...</td>
      <td>Nike Air Max SYSTM DM9537-001 https://tidd.ly/...</td>
      <td>https://www.youtube.com/watch?v=wKEF7-LNhSo</td>
      <td>RELEVANT</td>
    </tr>
    <tr>
      <th>10</th>
      <td>hb0j9Qn-KjM</td>
      <td>5 Craziest AI Agents We've Ever Built</td>
      <td>🚀 Uncover the 5 Craziest AI Agents We've Ever ...</td>
      <td>https://www.youtube.com/watch?v=hb0j9Qn-KjM</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M-XfdqPBAMU</td>
      <td>HUAWEI Mate XT Hands-on &amp; Quick Review: Huawei...</td>
      <td>That's my dream foldable phone right now.\n#hu...</td>
      <td>https://www.youtube.com/watch?v=M-XfdqPBAMU</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


# Pre-processing Comments<a class="anchor" id="third-bullet"></a>

In this step, I made a function that reads comments from a video and save them as a list of dicts:


```python
def get_comments(video_id):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100000
    )
    response = request.execute()

    # creating a list
    comments = []
    for item in response['items']:
        comment = {} # and a dict to store the results
        comment['video_id'] = item['snippet']['videoId']
        comment['channel_id'] = item['snippet']['channelId']
        comment['author_name'] = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
        comment['author_channel_id'] = item['snippet']['topLevelComment']['snippet']['authorChannelId']['value']
        comment['like_count'] = item['snippet']['topLevelComment']['snippet']['likeCount']
        comment['total_reply_count'] = item['snippet']['totalReplyCount']
        comment['published'] = item['snippet']['topLevelComment']['snippet']['publishedAt']
        comment['updated'] = item['snippet']['topLevelComment']['snippet']['updatedAt']
        comment['text_display'] = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comment['is_op'] = 1 if comment['author_channel_id']==comment['channel_id'] else 0
        comments.append(comment)

    return comments
```


```python
# testing the function
video_id = '0hwCp1TSHAQ'
get_comments(video_id)[1]
```




    {'video_id': '0hwCp1TSHAQ',
     'channel_id': 'UCw0leRmeaX7R_9BDd3TvCzg',
     'author_name': '@procuradehorizontes',
     'author_channel_id': 'UCAqFlxxVe6EvYloBZYOMU1Q',
     'like_count': 0,
     'total_reply_count': 0,
     'published': '2024-10-05T21:42:00Z',
     'updated': '2024-10-05T21:42:00Z',
     'text_display': 'cara, muito obrigado. Muito fácil e simples.',
     'is_op': 0}



Now, we should decide which data fields we are going to work with. In principle, considering our main goals, the following fields seems to be enough:
- **ID:** video ID
- **Channel ID:** useful to compare with the 'Author Channel ID'
- **Author name:** can be used as an identifyer for each user, as well as the 'Author Channel ID'. One of them might be discarded later
- **Author channel ID:** can also be used as an unique user identifyer
- **Like count:** useful to map which comments are the most relevant or popular
- **Reply count:** also can be used to measure the comment's level of popularity
- **Published:** can be used to perform an analysis of comment publishing over time
- **Updated:** can be used as a real timestamp for the comment's content. For instance if a comment is originally negative and then the user changes it's content drastically, the code will read and consider the new content along with this new date, instead of the original published date.
- **Text Display:** the main content, which will be the scope of our analysis

The Channel ID refers to the uploader's channel, and the Author Channel ID is from the user that made the comment. So if they are the same, it means the comment is from the uploader/owner of the video. This might be useful if we want to consider them into our analysis or not!

In order for us to build and test our NLP pre-processing pipeline, let's use the previously obtained dataframe 'df_videos'


```python
# obtain comments for each of the videos and save in a dataframe
def get_video_comments(df_videos):
    all_video_details = []

    for index, row in df_videos.iterrows():
        video_id = row['video_id']
        video_relevance = row['relevance']
        
        if video_relevance != "NOT RELEVANT":
            comments = get_comments(video_id)

            # append the video's details along with comments to the list
            for comment in comments:
                all_video_details.append({
                    'video_id': row['video_id'],
                    'video_title': row['video_title'],
                    'video_description': row['video_description'],
                    'video_relevance': video_relevance,
                    'text_display': comment['text_display'],
                    'channel_id': comment['channel_id'],
                    'author_name': comment['author_name'],
                    'author_channel_id': comment['author_channel_id'],
                    'like_count': comment['like_count'],
                    'total_reply_count': comment['total_reply_count'],
                    'published': comment['published'],
                    'updated': comment['updated'],
                    'published': comment['published'],
                    'is_op': comment['is_op']
                })
                
    df_videos_comments = pd.DataFrame(all_video_details)
    
    return df_videos_comments

df_videos_comments = get_video_comments(df_videos)

```


```python
df_videos_comments
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>video_title</th>
      <th>video_description</th>
      <th>video_relevance</th>
      <th>text_display</th>
      <th>channel_id</th>
      <th>author_name</th>
      <th>author_channel_id</th>
      <th>like_count</th>
      <th>total_reply_count</th>
      <th>published</th>
      <th>updated</th>
      <th>is_op</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8y5-UvxoD2E</td>
      <td>You’re wrong about Nike Air Max 270</td>
      <td>If you're ever injured in an accident, you can...</td>
      <td>RELEVANT</td>
      <td>If you’re ever injured in an accident, you can...</td>
      <td>UCId9g4zlQ9BOn6fLKIt1Y0A</td>
      <td>@RoseAnvil</td>
      <td>UCId9g4zlQ9BOn6fLKIt1Y0A</td>
      <td>33</td>
      <td>7</td>
      <td>2023-06-30T20:26:50Z</td>
      <td>2023-06-30T20:26:50Z</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8y5-UvxoD2E</td>
      <td>You’re wrong about Nike Air Max 270</td>
      <td>If you're ever injured in an accident, you can...</td>
      <td>RELEVANT</td>
      <td>&amp;quot;hotrods&amp;quot; don&amp;#39;t have the engine ...</td>
      <td>UCId9g4zlQ9BOn6fLKIt1Y0A</td>
      <td>@bennywolf2169</td>
      <td>UCU2xViJic33PhfJVezQLOgw</td>
      <td>0</td>
      <td>0</td>
      <td>2024-10-15T23:00:26Z</td>
      <td>2024-10-15T23:00:26Z</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8y5-UvxoD2E</td>
      <td>You’re wrong about Nike Air Max 270</td>
      <td>If you're ever injured in an accident, you can...</td>
      <td>RELEVANT</td>
      <td>former air max 270 owner here (the sole just r...</td>
      <td>UCId9g4zlQ9BOn6fLKIt1Y0A</td>
      <td>@tatumvp0</td>
      <td>UCBaeLavviiBLlPOOri5TXsg</td>
      <td>0</td>
      <td>0</td>
      <td>2024-10-12T07:28:46Z</td>
      <td>2024-10-12T07:28:46Z</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8y5-UvxoD2E</td>
      <td>You’re wrong about Nike Air Max 270</td>
      <td>If you're ever injured in an accident, you can...</td>
      <td>RELEVANT</td>
      <td>Personally the modern athletic shoe( Nike and ...</td>
      <td>UCId9g4zlQ9BOn6fLKIt1Y0A</td>
      <td>@Dukeofmamucas</td>
      <td>UC15XHBYm-k7Ol7bEafuLRig</td>
      <td>0</td>
      <td>0</td>
      <td>2024-10-10T17:50:16Z</td>
      <td>2024-10-10T17:50:16Z</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8y5-UvxoD2E</td>
      <td>You’re wrong about Nike Air Max 270</td>
      <td>If you're ever injured in an accident, you can...</td>
      <td>RELEVANT</td>
      <td>They nice shoes but I use to wear mines for se...</td>
      <td>UCId9g4zlQ9BOn6fLKIt1Y0A</td>
      <td>@markjuarbe1048</td>
      <td>UCwuPHypXuIlJlzzR3YV-ONQ</td>
      <td>0</td>
      <td>0</td>
      <td>2024-10-09T00:04:15Z</td>
      <td>2024-10-09T00:04:15Z</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>949</th>
      <td>M-XfdqPBAMU</td>
      <td>HUAWEI Mate XT Hands-on &amp; Quick Review: Huawei...</td>
      <td>That's my dream foldable phone right now.\n#hu...</td>
      <td>NaN</td>
      <td>Watching with my Samsung s20 ultra 😂 next upgr...</td>
      <td>UC_alSLqNTLW624x66BLgkHg</td>
      <td>@jesusloveyou9523</td>
      <td>UC3fTPEUTT_z6oRbQrkZiTKw</td>
      <td>0</td>
      <td>0</td>
      <td>2024-09-17T16:37:03Z</td>
      <td>2024-09-17T16:37:03Z</td>
      <td>0</td>
    </tr>
    <tr>
      <th>950</th>
      <td>M-XfdqPBAMU</td>
      <td>HUAWEI Mate XT Hands-on &amp; Quick Review: Huawei...</td>
      <td>That's my dream foldable phone right now.\n#hu...</td>
      <td>NaN</td>
      <td>F Apple with BS marketing !</td>
      <td>UC_alSLqNTLW624x66BLgkHg</td>
      <td>@joekerr8037</td>
      <td>UCrsKd9xCjAVAhM6ps1P6EFA</td>
      <td>0</td>
      <td>0</td>
      <td>2024-09-17T15:19:28Z</td>
      <td>2024-09-17T15:19:28Z</td>
      <td>0</td>
    </tr>
    <tr>
      <th>951</th>
      <td>M-XfdqPBAMU</td>
      <td>HUAWEI Mate XT Hands-on &amp; Quick Review: Huawei...</td>
      <td>That's my dream foldable phone right now.\n#hu...</td>
      <td>NaN</td>
      <td>Now imagine the US didn&amp;#39;t sanction Huawei....</td>
      <td>UC_alSLqNTLW624x66BLgkHg</td>
      <td>@xxBlackpspxx</td>
      <td>UCWomqdo5j4hLbqz22on3vYg</td>
      <td>0</td>
      <td>0</td>
      <td>2024-09-17T14:49:12Z</td>
      <td>2024-09-17T14:49:12Z</td>
      <td>0</td>
    </tr>
    <tr>
      <th>952</th>
      <td>M-XfdqPBAMU</td>
      <td>HUAWEI Mate XT Hands-on &amp; Quick Review: Huawei...</td>
      <td>That's my dream foldable phone right now.\n#hu...</td>
      <td>NaN</td>
      <td>There is a huge gap between some &amp;quot; materi...</td>
      <td>UC_alSLqNTLW624x66BLgkHg</td>
      <td>@向日葵向日葵-c4v</td>
      <td>UCd4Cmtp0-wSQ0ihhxyicAug</td>
      <td>0</td>
      <td>0</td>
      <td>2024-09-17T14:08:34Z</td>
      <td>2024-09-17T14:08:34Z</td>
      <td>0</td>
    </tr>
    <tr>
      <th>953</th>
      <td>M-XfdqPBAMU</td>
      <td>HUAWEI Mate XT Hands-on &amp; Quick Review: Huawei...</td>
      <td>That's my dream foldable phone right now.\n#hu...</td>
      <td>NaN</td>
      <td>创意不错</td>
      <td>UC_alSLqNTLW624x66BLgkHg</td>
      <td>@von4873</td>
      <td>UC5VFXd8Y4IJQfCtdvimx_2Q</td>
      <td>0</td>
      <td>0</td>
      <td>2024-09-17T13:26:49Z</td>
      <td>2024-09-17T13:26:49Z</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>954 rows × 13 columns</p>
</div>



Now the **text_display** column contains all our data to be analyzed, but in order for a wordcloud to be created from them, this data shall be pre-processed so the comments are clean and in a format that is easier to be used by models and other functions.

First, in our pre-processing, we shall determine what actions should be done and in which order to obtain the best results.

From previous experiences in other projects, in this pre-processing pipeline the following steps should make the most sense:
- Remove numbers
- Remove mentions (contains @)
- Remove links
- Demojize (convert emoji into words with their meaning)
- Standardization of the most common keyboard emoticons
- Remove special characters
- Convert all text to lowercase
- Standardize common slangs/expressions
- Removal of characters that repeat 3 or more times within a single comment
- Tokenize (splitting the string into tokens - the smallest string blocks that can be considered)
- Stopwords removal
- Lemmatization (convert words into their "root" form - variations of the same are grouped up)

We shall import the 'nltk' (Natural Language Toolkit) lib, which includes tools for text processing, classification, tokenization, stemming, and much more... as well as 'emoji' lib, 're' (regex), and 'unidecode' to treat accentuation.


```python
#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('wordnet')
```


```python
# stopwords are a pre-defined collection of words that generally don't add any significant semantic meaning to text,
# so it's a common practice to remove them

stopwords = nltk.corpus.stopwords.words('english')
stopwords = list(set([unidecode(word) for word in stopwords])) # remove accentuation

lemmatizer = WordNetLemmatizer()

print(stopwords)
```

    ["didn't", 'for', 'having', 'couldn', 'these', 'if', 'under', 'own', 'very', 'i', 'have', "you'll", 'into', 'shouldn', 'until', 'most', "that'll", 'or', 'about', "it's", 'are', 'all', 'of', 'few', "haven't", 'it', "she's", 'your', 'too', 'by', 'each', 'aren', 'against', 'that', 'hers', 'shan', "don't", 'above', 'on', 'themselves', 'mightn', 'a', "needn't", 'once', "weren't", 'itself', 'our', "you're", "shan't", 'be', 'through', 'he', 'some', "hasn't", 'then', "hadn't", 'when', 'weren', 'them', 'than', 'won', 'same', 'did', 'was', "shouldn't", 'there', 'we', "mustn't", 'down', 'only', 'where', "couldn't", 'him', 'between', 'yours', 'me', 'out', 'himself', 'whom', 'should', 'with', 'haven', 'but', 'they', 'is', 'the', 'as', 'such', 'being', 'here', 'ain', 'yourselves', 'after', 'more', 'my', 've', 'no', 'so', 're', 'isn', 'do', "aren't", 'because', 'were', 'doing', 'ma', 'her', 'what', 'yourself', 'while', 'further', 'nor', "won't", 'before', 'o', "wasn't", 'hasn', 'doesn', 'needn', 'and', 'his', 'mustn', 'other', 'their', 'below', 'm', "you'd", 'an', 'those', 'will', 'over', 'to', 'who', 'am', "should've", 'any', 'can', 'you', 'not', 'why', 'from', 's', 'now', 'd', 'just', 'ours', 'off', 'has', 'had', 'y', "mightn't", 'wouldn', 'again', 'during', 'hadn', "doesn't", "wouldn't", 'ourselves', 'wasn', 'which', 'she', 'theirs', 'its', 'at', 'didn', "isn't", 'don', "you've", 'herself', 'this', 'myself', 'been', 'how', 'up', 'both', 't', 'll', 'in', 'does']
    

After some testing with the code below, I noticed two things: first, that there are some candidates for stopwords that weren't considered in the original list but they could be included. Second, some comments are in different languages other than English, and that surely needs to be dealt with in order for our pre-processing pipeline and LLM mdoel to work properly.

So we shall use another lib to identify the language, and only English will be considered at this time because otherwise we would need to build a separate pre-processing pipeline for each language. And since we don't know how many different ones we could find, this would be a project enhancement to be considered in the future.


```python
# Declaring a seed
DetectorFactory.seed = 42

# Function to detect the language
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'
```


```python
# applying language detection to each comment
df_videos_comments['language'] = df_videos_comments['text_display'].apply(detect_language)
```


```python
# many languages were detected
df_videos_comments['language'].unique()
```




    array(['en', 'de', 'tl', 'sl', 'pl', 'so', 'fr', 'ko', 'et', 'af', 'no',
           'fi', 'sw', 'ro', 'id', 'unknown', 'nl', 'it', 'ne', 'lt', 'tr',
           'es', 'pt', 'cy', 'da', 'ca', 'hr', 'ru', 'hu', 'sk', 'sv', 'lv',
           'bg', 'zh-tw', 'zh-cn'], dtype=object)




```python
# checking percentage of comments in english
len(df_videos_comments[df_videos_comments['language']=='en']) / len(df_videos_comments)
```




    0.850104821802935




```python
# only consider english comments
df = df_videos_comments[df_videos_comments['language'] == 'en']
```


```python
# adding more stopwords found after some testing
stopwords.extend(['href', 'quot', 'br', 'u', 'r', 'lt', 'b'])
stopwords = list(set([unidecode(word) for word in stopwords]))
#print(stopwords)
```


```python
# subfunction to idenify emoticons made with keyboard keys
def find_emoticons(string):
    happy_faces = [' :D',' :)',' (:',' =D',' =)',' (=',' ;D',' ;)',' :-)',' ;-)',' ;-D',' :-D']
    sad_faces = [' D:',' :(',' ):',' =(',' D=',' )=',' ;(',' D;', ' )-:',' )-;',' D-;',' D-:',' :/',' :-/', ' =/']
    neutral_faces = [' :P',' :*','=P',' =S',' =*',' ;*',' :-|',' :-*',' =-P',' =-S']
    for face in happy_faces:
        if face in string:
            string = string.replace(face, ' happy_face ')
    for face in sad_faces:
        if face in string:
            string = string.replace(face, ' sad_face ')
    for face in neutral_faces:
        if face in string:
            string = string.replace(face, ' neutral_face ')  
    return string
```


```python
# pre-processing function - pipeline
def preprocessing(string):
    # remove numbers
    string = re.sub(r'\d', '', string)
    # remove mentions
    string = re.sub(r'\B@\w*[a-zA-Z]+\w*', '', string)
    # remove links
    string = re.sub(r'http\S+', '', string)
    # demojize
    string = emoji.demojize(string)
    
    # padronização de emotes
    string = find_emoticons(string)
    
    # remove accentuation
    string = unidecode(string)
    
    # remove special characters
    string = re.sub(r"[^a-zA-Z0-9_]+", ' ', string)
    
    # lowercase
    string = string.lower()
    
    # padronização de risadas
    string = re.sub(r'\w*haha\w*', 'hahaha', string)
    string = re.sub(r'\w*lol\w*', 'hahaha', string)
    
    # remove repeated characters (more than 3 times in a sequence)
    string = re.sub(r'(\w)\1(\1+)',r'\1',string)
    
    # tokenization
    words = word_tokenize(string)
    
    # remoção de stopwords
    filtered_words = []
    for w in words:
        if w not in stopwords:
            filtered_words.append(w)
    
    # lemmatization
    lemma_words = []
    for w in filtered_words:
        l_words = lemmatizer.lemmatize(w)
        lemma_words.append(l_words)
        
    return lemma_words
```


```python
df['text_display'].iloc[0]
```




    'If you’re ever injured in an accident, you can check out Morgan &amp; Morgan. Their fee is free unless they win. For more information go to <a href="https://www.wordontheblock.biz/r/2369/86/?s=FOR_THE_PEOPLE">https://www.wordontheblock.biz/r/2369/86/?s=FOR_THE_PEOPLE</a>'




```python
preprocessing(df['text_display'].iloc[0])
```




    ['ever',
     'injured',
     'accident',
     'check',
     'morgan',
     'amp',
     'morgan',
     'fee',
     'free',
     'unless',
     'win',
     'information',
     'go']




```python
%%time

df['text_filtered'] = df['text_display'].apply(lambda x: preprocessing(x))
df['text_joined'] = df['text_filtered'].apply(lambda x: ' '.join(x))
```

    CPU times: total: 141 ms
    Wall time: 143 ms
    

    <timed exec>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    <timed exec>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    


```python
for _ in df['text_joined']:
    print(_)
```

    ever injured accident check morgan amp morgan fee free unless win information go
    hotrods engine back hahaha
    former air max owner sole ripped year use never really comfort issue greek foot shoe ever sightly big would agree though comfortable deflated biggest ick
    personally modern athletic shoe nike adidas supportive upper
    nice shoe use wear mine security work oh man foot pain
    bought look cool hell comfy shoe ever worn nah even close adidas cloudfoam shoe mile comfy even talking ultrabounce similar air max mile better
    air max v air max trying show son better also cost ridiculous get
    dude litterly went dick sporting good stood bought
    jordan mar real shoe
    anyone interested bought came liked immediately changed last year bought white became gray time sticker therefore number started come time plastic sole started become hard uncomfortable last washing often shoe shrunk raised thumb made hole shoe year shoe lasted longest
    fake one
    nike air never performance gimmick start
    pretty sure comfort subjective owning pair perfect traveling week around k step day
    correct review feets hurt wearing nike opinion nike react vision x comfortable fact
    shoe bro
    literally everybody different perspective personal experience comfortable wore multiple theme park held well even wore pouring work outside still held well comfortable thing noticed run little slim year wearing small hole great shoe opinion respect
    one worst pair shoe ever bought uncomfortable huge drop heel toe tight restrained fit toe hit front mesh time ache like hell thought initial break shoe problem hence return got worse give pain shin also still believe spent much give horror foot sad_face
    nike worst trainer made still cant wear side feel every single bump stone etc underfoot ive trying wear house trainer min im done lasted two trip side chucked bin look fab thats going thankfully purchased pair
    aprecciate thanks buy
    everyone wrong right got
    love air max durable went pair
    one like shoe wear awful
    got jordan mar cheap impressed heel unit seems like style substance seems add cushioning beyond naturally apply pressure
    feel like walking heel wear crap
    jordan mar muck better shape comfortable air unit
    comfortable comfortable opinion crocs comfortable think
    adidas guy recently owned pair nike since suffered plantar fasciitis heel pain past month went looking new sneaker air max calling made purchase soon start walking plantar pain go away adidas gazelle always comfortable sneaker air max taken top spot
    quebec fault law sue driver accident adopted rene lesvesque era strangely semingod politician alcoolic nearly killed mistress drunk driving incident
    think need inc size
    good review thanks lot
    appreciate review method use test air max arch support convex position shoe important thank
    like lot tbh
    way dude slope running shoe even airmax way steeper maybe wear long enough ridiculously comfortable especially time terrible uneducated review
    th clip seen assessment finalise yes know foot comfort looking boot thumbs_up handshake
    buy heel drop stupid felt straight away
    really enjoy see talking toe front issue considering toe even broke
    totally agree comfortable experiencing pain around middle area foot
    sneaker head see get half drop dirty water jordan
    amazing stuff saved money amp wife foot like people got caught look amp forcing place order size amp stock run thank fully video saved amp keep great work
    wow finally someone telling like happens lot nike shoe another example nike invincible run every reviewer swears comfortable reality one uncomfortable shoe ever worn great running unstable completely unwearable casually yeah gaslighting real nike world
    agrre everything say bought cause like lot uncomfortable incline step like forced run walk fust still feel tight forefoot area try slimer sock walk get used feeling dont know might give away nice shoe design
    thing made people wide foot personally love
    disagree buddy definitely comfortable
    got fake one
    initially felt odd insole design however decent break fell right form make think knew would happen designed form hell idk hahaha
    purchased height boost skull
    seriously even tried pair really comfortable heel drop air unit way high point like people still able walk zoom air athletic purpose original air max
    pair hate wish never bought face_with_tears_of_joy super tight inside foot along super tight heel odd slope design heel shoe suck get real air max shoe actually feel supposed feel like
    couldnt agree pair first thing noticed toe side soul like lot slope trying tilt forward toe end get slim almost feel every little grain ground toe yeah bottom sole horseshoe shaped rubber bubble came
    mean toe correct false consensus biased assuming everybody physiology
    air max would recommend everyday walking thank
    great review bang look benjamin nike smirking_face upside down_face
    sorry pair comfortable
    opinion literally comfortable sneaker ever owned
    fake made vietnam real one made indonesia
    probably late air max
    find good walking shoe weight heel im walking pretty shit running walking cloud vista running ever said best running straight cap trried running uncomfortable stable actually tore something running first time becuase huge bubble slamming front foot run heel toe
    thought got foot injured week wearing volleyball smiling_face_with_tear wish watched buying expensive
    week weird degree angle put lot pressure heel ended returning
    called shoe new iphone
    ego guy believe think comfortable therefore lying
    man european mm inch comparison nastie okay zero idea inch mean could guess say running shoe idk inch drop could thought okay difference mm inch bocsanat hell figure difference grinning_face_with_sweat face_with_tears_of_joy
    amen hate shoe immediately hated soon put heel stick way far definite placebo effect cut open pair asics kayanos year ago kayano also scam product late party love see pair nike footscapes cut open exploding_head
    biggest air bubble dont even stand crazy
    used sneaker guy high school wear nb dead still comfortable shoe ultraboost foam last incredibly long real shame increase thickness tread could wear year wear
    hated
    bro pair air max shit even moderately comfortable loudly_crying_face
    shoe straight tras last awful fake bubble even anything
    great review bang worst thing ever put foot much prefer brook glycerine bought look fuction brigade
    owned nike trainer year version air max far best range easily
    channel useless want test shoe wear weird test using metal ball hell man waste air
    really bad foot cant wear thin sole shoe without foot hurting either wear adidas alpha bounce nike something shoe make super comfortable
    tried show yr old daughter want pair stopped wearing mine yr cuz uncomfortable one basic friend think cool think video testimonial wrong forgot know everything
    always pretentious douche vibe homeboy video
    mile universal studio one complaint
    guy went army infantry training surprised survived
    im struggle thats narrow foot
    nike fan care brand know shoe idea shoe reputation buy shoe look nice comfortable bought since tried store used work walking ton maintained level comfort plan keep long time began squeaking exchanged replacement hopefully next one squeak feel forward sliding wear day feel consistent otherwise returned using month
    hot rod engine back man_facepalming_light_skin_tone rolling_on_the_floor_laughing
    still think nike free rn flyknit comfortable casual athleisure shoe
    air max would great test popular sneaker far air max line concerned
    make review lebron witness
    hideous shoe
    love functional fashionable slip ons imo air max gnome popped air unit favourite sneaker hahaha
    get size real size got feel pretty good foot
    collect runner amazing find comfortable day one break day comfy never popped one many pair also wear ultraboost asics amazing well
    exactly feel like walking downhill
    see coming people foot built different depending person may comfortable people similar foot structure least
    worst nike ever per wear tear plus fact time basically feel air unit foot far forward comparison recent air max footwear purchase vapor max scorpion fking worst waste money
    bottom half shoe cool plain top shoe design shoe
    grey orange love comfortable
    air max head brainer honestly sleek silhouette great new model good one
    great review one hand comfortable tn dn
    fan air max one time step small nail shoe got messed try ti put plug bit work grinning_face_with_sweat
    favs airmax plus comfortable
    air max favorite nike shoe line moment one wild
    wait next year model
    anyone tell model air max
    like look cool leisure shoe anyway
    looking new pair kick summer almost got another pair pair love feel light nimble seeing max dn s went sky blue variant wait get
    anyone gotten hole creased area shoe
    model best ever ordered one tell replace
    leave la gon na fulfill order
    hurt foot
    dn day sick he got night
    would take new balance
    really light
    upper mine ripped six week shoe crease walking pure cheaply made garbage
    fell people deemed offensive nike kinda stopped selling suck oh well
    got went size use gym brilliant jogging comfortable work well
    rocked adidas alphabounces airmax day
    comfy running shoe much swapped new runner
    thought shoe light made commercial lighting find nope
    light whack
    ordered cream colored one wait get see
    got junior size label back say air max etc air bubble normal
    remind alot air max plus
    love air bubble want grey pair
    expensive careful buy got toe pain dn comfy like said opinion think twice buy
    light thumbs_down_medium dark_skin_tone
    nice glowed
    best time personal taste pioneer air max air max bw never stop making nice presentation
    thouth led light inside air max cushion shame sad_face
    hahaha wear time waiting bubble burst nike quality ftw
    want like nike literally every pair nike shoe uncomfortable shoe ever look comfy though
    bruh get better sock skull skull make shoe look worse know dress
    one ugliest air max shoe ever see definite
    love black pink one size wear kid size kid size different especially back bit look carbon fibre kid one different part might pay extra adult one
    true size better order half
    people realize two different psi connected unit let get right high psi last two bubble move front lower psi front two bubble move ball foot air go back two different psi around call magic c mon ppl society really dumber
    surprised positive feedback model look like ugly hybrid deluxe sitting right next tn pimento one best colourways one best air max model imo suck even maybe european thing love tn
    bro wait see shoe dbz custom dragon ball air capsule
    want vapormax auto lace feature able made nike id site mine melted leg got blown ive never able find comparable shoe
    two comfortable model day wearing
    uncomfortable trainer ever bought loved look felt good shop hour pain toe knuckle bad loved trainer tried stretching help suffer wide foot avoid peril problem air max disappointed really love
    came hoping reviewed knew would thanks content fence passing pro amp con flat foot mess narrow shoe also mostly performance shoe kinda guy love techy stuff fore foot air bag also negative point moving tongue spot tn one greatest silhouette ever drawn definitely could used strap gusset keep stable glad still making video dude great night bet harrison huge face_with_tears_of_joy
    site get air max dn
    think canada right suck man really wan na check hahaha oh well
    air max sick wow man love sure one order get access
    love order kick like get access site reason get access say denied
    telling access page denied
    shoe sick man wow
    upload k
    look nice love black whhite one going yo get pair love air max sneaker comfortable red_heart red_heart red_heart
    need bigger swoosh maybe even jeweled one color sick though
    purchased color way even better hand hype aside air max head fence like since release day pull trigger happy need get white color way summer smiling_face_with_heart eye
    look like hot air max shoe summer
    find anywhere
    anything air max plus black chrome volt concord metallic silver released seen video yet
    go running shoe
    saying foot shoe given seen shoe youtubers say like saying shirt look great torso hat look great head
    got exact pair although hoping get pair cocoanut colorway unfortunately cocoanut sold size hoping restock soon since really fan black shoe see wearing black crimson often personally still like scorpion air max made date hell like scorpion much bought every colorway made men even bought pair womens lmfao
    wish could afford
    shoe definitely good looking good design overall couldve branded nocta shoe definitely something drake wear term new airmax tech feel like marketing gimmick
    anyone happen know fit u people wide foot
    think look little cheap like walmart kick knock offs probably grow still prefer
    bubble actually light tho
    got blue white
    question black pair reviewing video available woman love colour available men crying_face
    air max die hard would love nike incorporate react cushioning air pod would redesign come ankle lix fit feel supportive stretchy mesh
    thing look stupid lmao look like tonka truck rear wheel
    think new model want mass produce would benefit hugely day nikeid release would certainly push paying pair getting something know going like
    cushioning forefoot also boost bubble bounce whatever forefoot
    look like hold extra materia final fantasy
    think air max black neon green better choice buy
    let get straight made whole video right made comparison every shoe except one closest resembles airmax pulse right
    dont like newer model bulk purely midsole upper sleek like upper bulk comfortable shoe especially current colorways
    definitely style im vibing im getting pair mix jordan set might pick pop airmax shox running shoe kina guy
    idk long take break god damn foot killing one day use walking around
    look horrible technology great
    got sneaker month ago grabbed release think super comfy also new sneaker world got color really enjoy far
    um feelin pick royal big bubble clean hell
    got pair dn suggest sizing first make sure measure true size according nike size chart choose true size go go size true size fit perfectly party_popper
    getting mine tomorrow
    black gold red_heart
    need color like black gray
    bro bought last edition annual air max release fall quickly nothing better air max
    tube connected rear psi tube connected air tunnel psi tube connected air tunnel look like one piece physically joined piece keep one place
    exact colorway arrived mail today super comfortable break period definitely order half size better comfort air pod thing gimmick least feel foot far pretty stable heel section really mushy
    l like trainer fabric back trainer grip colour tht horrible air unit
    stability like tw pair roll ankle step anything uneven
    agreed air max greatest
    think forefoot zoom air also sale soon already sale japan store
    going jog air force one
    bought
    hm consider trying nooky pookies offer many complex simple unique desings come bearfoot shoe removable high heel casual shoe check em today
    hate dirty mind loudly_crying_face loudly_crying_face loudly_crying_face think thing everytime said dick
    send one pair face_with_tears_of_joy
    love shoe red_heart kash mein eshe khareed pati
    wear shoe run store without paying
    give gift shoe sleepy_face
    bro skull
    brother please give pair shoe gift want prepare police duty
    imagine kicked dog shit trying clean inside front side shoe sole
    price crying_face crying_face crying_face crying_face crying_face crying_face crying_face crying_face crying_face high afford
    annoying short time stamp
    shop name skull skull skull
    making shoe appealing eye going next shoe feel like jumping mattress
    durability joke
    didnt know could go store make vid stuff dont
    need suspension way push
    many price
    want exact colour way gorgeous
    much shoe
    dick reason called dick face_with_tears_of_joy face_with_tears_of_joy
    want pair workout
    paid influencer
    please order
    need shoe please gift send thumbs_up_light_skin_tone thumbs_up_light_skin_tone
    much cost india price
    wan na run fast cheat
    look cheap
    run store
    buy one add another degree original size like size shloud buy
    going
    wish could afford disappointed_face
    buy nike store anymore crying_face
    price shoe plese tell
    rip
    wrong decision referee
    purchase shoe
    btw want run fast train harder change shoe sure faster anyone running barefoot
    worth wouldnt change converse chuck taylor model
    bro hard shoe cant fine shoe
    diddy test dick
    shoe much nicer looking trainer brand wish could make version altered slightly training purpose
    price kya hai
    wan na buy get
    give gift smiling_face_with_smiling_eyes red_heart
    many piece indian rupee
    nike running shoe trash face_with_tears_of_joy
    like shoe help money buy shoe
    many price shose pls answer pls pls
    buy ur short
    shoe cut bullshit seriously bullshit get worst day day
    price
    price kya hai
    running shoe cost dollar
    us walking shoe
    shoe good flat footed people
    much money
    please brother one shoe gift please mony problem hai please
    every single time break certain spot front shoe next symbol love shoe always break anyone advice
    really snock
    shoe men woman
    isthe best
    men version look identical womens love seeing woman wear im make self conscious hahaha
    shoe wrkout gym style
    really
    hey got wider foot get half size higher actual fit get actual size
    long usually last
    buyed shoe basketball shoe spike
    awesome review man saw sale recently video made buy em
    bought used decent pair drawback air bubble really scuffed tip buff make look almost new
    got work im standing day praying good fit grinning_face_with_sweat returned cloudtech bc one arch heel get used ball foot hurting half way shift
    true give much height unbelievable highest shoe ever worn
    got first high top shoe year old year cocky wearing gym shoe life thanks mom taking local shoe shop running aside feel like thankful come shoe
    play basketball
    would recommend playing basketball
    face_with_open_mouth best nike face_with_open_mouth face_with_open_mouth
    use snow
    buy summer live realy hot
    got question bro got mine feel like im walking edge insole better time normal
    sir much better look airmax nike sb nyjah
    call two seven
    nike air max alpha trainer
    bought pair thinking going back next month get nd pair set next year buy nike shoe every year
    good sport play sport quickly move different direction need something handle quick directional movement
    never really owned shoe bubble good outdoor running workout recommend
    anyone answer comparison question v air max se curious anyone tried pair air max se bought one pair realized comfortable truly bought pair hahaha never shoe comfortable want try pair solid
    size eu u air unit right shoe literally got punctured month indoor use fortunately company bought replace wear hour day job really say comfortable review say better option would nike zoom shoe looking bouncy sole one recommend allt feel cheap overall
    amazing review earned another subscriber got gift look great agree wear feel alot taller half inch much liking since take like wtf workout sometimes hahaha also defined agree go half size take bit first get comfortable get used
    funny bro probably favorite shoe reviewer hundred_points
    bought black neon went sale eur bought school casual event hopefully last
    really enjoyed simple concise comprehensive review thank much love
    wear work standing day walking going place
    year good shoe air bubble left started leaking practically unusable take care em think love em
    size galaxy black white white amazing shoe everything
    foot normal get regular size
    good flat foot wide foot got flat foot
    ok_hand_medium dark_skin_tone bought year old son
    nike shoe like wide foot wear adidas wear em month basketball street asphalt work
    good people heavy
    compare prestos fence
    great personality watch
    best color long term
    probably greatest air max far
    love white hot pink neon orange neon yellow want black white find extremely comfy white biatch keep clean hate dirty shoe
    one best review seen straight point stalling minute subbed
    pinkish kinda color bottom change certain month like get different color shoe want
    took trip london past weekend walked k step minimum day practically lived super comfortable added bonus look fire
    got black white black purple love im worry connection unit sole back look like peeling need reglued month drummer tho play mibbe reason dying little faster fresh af slorps tho
    currently pair getting like pair th birthday soon
    got school air max beat bought one aud bought
    bottom blad red fake
    thing pain as put dope af sneaker overall smiling_face_with_sunglasses
    omfg lmao doug demuro shoutout loved
    got birthday
    ima get birthday exited thx telling feeling get used
    designed tongue hard angle ankle moving lace change nothing hurt
    wearing carpet smiling_face_with_tear
    measure height without tell height boost nearest eight inch
    pair arrived today absolutely love wished pink bottom bright rolling_on_the_floor_laughing
    think imma buy kid version foot tiny dollar cheaper take price cut lord pricey
    hello question im sure colour get black white one black one red black one buy tried store im size tried size fit great one thing noticed step shoe side fold tie shoe tight enough thank
    curiosity cop pant wearing around second mark
    awesome video friend one question kid size difference grade school orginal
    god loved doug demuro reference face_with_tears_of_joy face_with_tears_of_joy face_with_tears_of_joy super nice review detailed straight point also loved style reference end ordered one pair day
    good running
    hey know late make video tie new balance like make look good
    plan reviewing afs
    like video man nice meet regard colombia colombia
    heresy wear jean
    live white one look great
    use code tester save poizon sneaker luxury good
    comfy tho
    care extra white letter much white taste pair og remember paying back worked local bk cali face_with_tears_of_joy good review chris victory_hand
    bought pair today
    keep bringing back everything except air ubiquitous max
    dope yes chris triple white must
    yall always fact checking face_with_tears_of_joy
    shoe look epic foot love em heal pillow kinda feel weird hurt sometimes shoe look cool smiling_face_with_sunglasses grinning_face_with_sweat
    fire got blue white olympic pair shoe palace fit tt comfortable retro
    luv old og new air ugly
    much nicer person saw today mall picked gon na beat ground face_with_tears_of_joy
    air jordan black toe
    look better low
    bother much aligned midsole deal breaker cool review way hard find shoe name thanks
    going back forth think grab bad
    really look like low mid
    joker cutscene actually made hahaha
    exactly first thought nike sign white silver would definitely made shoe stand still nice
    actually prefer ambush version coo
    agree initial wtf reaction certain color also made say kind dope
    funny certain j colorways get called grade school special yet always seem get pas dope like
    bad well done nike
    look kinda streamlined bulky sneaker like allot smiling_face_with_smiling_eyes smiling_face_with_smiling_eyes smiling_face_with_smiling_eyes
    shit fire black gum bottom joint
    future question day golden state warrior nba franchise favorite
    vybe depending colorway
    never liked huge air side much
    definitely ninja hahaha getting closer look bad expecting
    ok personally like ambush better ninja
    designer think current regime personally think much last year massive race bottom large company slowly diminishing product training customer accept less lust hype little substance behind think concept revisiting old style brilliant massive reebok fan think anytime ever done well perhaps billy hoyle really well done certainly best pump made anybody last year honestly look billy hoyle outlier happy_face
    look interesting qotd ninja
    thealgo
    ambush low first like thinking_face thinking_face thinking_face thinking_face might fw
    bulky sad_face
    wonder penny would play
    love way look actually made air fit lower profile look squished pair original played around bit clunky heavy built like pip face_with_tears_of_joy question day ninja day monochrome aesthetic anyway functional ninja_medium_skin_tone
    historically pirate tended egalitarian thus likely treated female better treated regular society
    would love ninja
    orignl gorgeous
    check sea shanty medley male vocal group home free
    awesome star struck
    reminds crazy made low recently dope hopefully see soon
    colorway fire idk one like shoe
    guy hamster get every time lmao
    ninja day contest going question would rather superhero anti hero
    fire look high love
    nike go back big air bubble yes ruined classic
    kinda feeling cause straight cut top nothing else
    want cw next
    got grey crimson
    nike probably making guy sign ndas cant even speak nike light could perceived negative
    please make uptempo k low
    ruined face_with_crossed out_eyes
    need white pair clean water
    like shoe always mixed feeling uptempo anyway depends day sometimes kinda like time would ninja question wanted pre teen year beaming_face_with_smiling_eyes
    ninja pirate get take shower dr mcninja snake eye decide
    uptempo might time favorite air sneaker low never needed think ambush way better job attempt though
    look good curious see person awesome review always fire ambush version forgettable even remember made one grinning_squinting_face
    could review puma pro nitro elite
    qotd ninja boat future qotd halloween costume year favorite thing ever gone og purist checking never felt like essential pair air uptempos personal preference think original design iconic messed appreciate review got like got comment great weekend see soon
    ninja yall remember ninja rocky love emily
    need zoom flight aka j kidd
    difference ambush pair
    sol black hate white bottom bit white keep nubuck mesh ugly
    love ninja seeing pirate cheat gun ninja history deadliest warrior always stuck meaning pirate cheating win thought coolest thing
    hey chris jodi great stuff usual long time viewer circa totally okay title video could guy put shoe name title end video harder find video fact several month year name whole shoe really visible thumbnail fine either two would amazing thanks everything
    team pirate sadly discipline pas ninja reckless like pirate see shoe like buy arrggh er pirate_flag
    younger brother original one black ok_hand man long time ago died least still jordan
    anne bonney grace malley could worth google ninja whose job spy nice house definitely sea
    adidas crazy low next clapping_hands_medium_skin_tone
    original good bad stil destroyed
    hate shoe always love people try new thing also ninja final answer
    like design get feeling j low
    fan
    honestly kinda like although prefer og actually pretty cool
    qotd encountered mythical creature yeti chupacabra loch ness monster screaming banshee would tell anybody
    ordered grey red one wait see already midtop ambush one black white would samy collection seeing vid black white look really clean might take another shot grinning_face_with_sweat
    really look like low much difference
    white pair coming home sure
    pippen used play pair air bubble painted said bubble used make ankle roll time werent truely meant performance
    enough nike stop messing around give u pippen play colorway
    actually look better mids like em smiling_face_with_sunglasses
    team ninja swift silent deadly
    low issue sale prove low sell way better mid high especially asia cuz ppl distorted perception high like inch boot thinkin make look really short bigger issue white fabric know really hard clean start get dirty
    ahoy ar uptempo low pirate_flag
    pirate thing would hate would long time sea
    buy rose anvil leather good everybody want
    ripstop terrible give like material shoe feel comfortable shoe ripstop experience
    watching video still bought pair
    say please return product received damaged product
    shoe cured insertional achilles tendonitis love
    find cutting edge shoe tech running sneaker everything else marketing gimmick
    shoe since came review also watched review year ago mine still new condiution everyone like day oh yes broken day comfortable especially foot injury ankle swapped insole orthafeet make feel like im walking air reason comment really im looking another pair decided look back reveiw thumbs_up
    flat foot thing fit like glove err like shoe
    air chamber heel psi middle less psi remember able get air chamber old nike without popping using barbaric method young want
    think middle air chamber separate pair like decade ago yes cut apart honestly thought weirdo needed cut old nike air open
    plastic junk thats nike
    mine since still good new time fresh pair would ever review nike air max pulse black anthracite ive looking day find replacement air max appreciate thought love cut even tho made cringe seeing cut hahaha thanks
    thank brother another great video curious see shoe made well show shoe back together fillowed step step need reverse thumbs_up thumbs_up thumbs_up
    expected much better durability airmax shoe got air max ltd year really careful walked concreted still got damaged edge inside glued air chamber shoe comfortable although first one shoe really uncomfortable finger reason got better time fault popped probably would buy another airmax pair like idea survive even half year
    great inner chamber connect outside bubble shoe popped year outside feel difference walking outside chamber look
    like see pair dc versatile cut half please wot difference nike air system
    nike air
    air
    got pair super comfy make inch taller happy_face
    least two separate chamber air bag one damage got another one backup imagine way somewhere want flat air max never put egg one basket basket
    guy burn cut open shoe despicable thing seen channel
    smiling_face_with_smiling_eyes
    want shoe air pass back forth walk reebok iverson answer dmx think retro real tech
    would love see teardown luxury brand leather shoe like zegna tod
    opinion wht worth shoe make toe crowd front unhealthy massive hazard wear rucksack feel rucksack end bumping people shoe shoe much wider foot really feel walk thing look either although love orange
    q always wonder much psi put inside
    cheaply made high end logo
    remember cutting old pair nike air cross country im talking probably im almost certain void outsole could well wrong memory strong point multiple brain damage
    got current air max model favourite air max motif comfortable one one know walking air clever sole design cheap p half price im lying soon
    exact shoe air unit tiring cut apart adidas printed shoe
    need airmax video folded_hands
    would awesome could source cut pair vintage trax shoe kmart
    air max video
    look cheap cheap toy shoe toy foot either get custom buy brand seems last longest cordless drill appeal
    always loved way airmax sneaker felt every single pair owned gotten hole leak ruin day force buy new pair sneaker
    please air max pre day think look wise kinda cool
    nike staff looking video take note
    damn shoe bad
    thing foot kinda want go front shoe heel tall definitely go away couple wear one first thing noticed got first pair
    mf spongebob scary gave heart attack real
    short king rolling_on_the_floor_laughing love guy much
    air max shit product
    ultra thin rubber outsole deal breaker seriously wear month term comfort bubble good chance popping like literally air max shoe nike good brand want zero foot pain hour run walk switched kayanos never looked back
    power mouth mouth flexed_biceps high_voltage television genjutsu marketing face_with_tears_of_joy technology bla bla bla face_with_tears_of_joy
    trust nike gloat recycled material use one component shoe wear thanks nike although realise centre bubble connected one applying pressure centre bubble cause squash pushing plastic strut others thereby compressing
    never buying nike fan since
    air max please smiling_face_with_tear
    wastage money
    brother
    top best selling budget boot amazon
    try cutting nike blazer half
    really need modulate cut content roll understand played need play crappy music volume change really friggin annoying adjust voice roll spike another db scrambling remote waking family face_with_open_mouth
    possible cut open nike waffle debut wondering legit cushioning shoe air unit zoom unit million thanks
    use shoe goo preserve sole
    great video taking time find truth shoe great see person large flat foot depth study definitely matter appreciate remember shoe first came colorway fresh book nike never mean never great regarding comfort started wearing orthotics young feeling pain foot nike understand comfort started wearing orthotics flatfoot late teen still grasped grasp style legendary history connects pop culture simply understand build comfort even orthotics comfort shoe still matter bubble feature shoe look appealing nothing eye candy give lasting comfort especially supposed bubble heel tell much anything guy said would seem heel pressure would disperse air within bubble underneath bubble visibly see give comfort heel land ground even effect look minimal giving anything cutting see truth every shoe eventually wear welcome use time shoe company often fool people aesthetic shoe really much gamechanger get older matter especially value comfort softer touch shoe personally wear nike anymore still appreciate great style colorways often distance go athletic shoe day tends asics built orthotic style sol flatfoot like asics gel kayanos others love see video done latest asics running shoe flatfoot runner gym body
    nobulls plastic bead shoe really protect shoe well
    hi music called
    air per say gas pressurized
    fake
    bro mad loudly_crying_face wanted color face_with_tears_of_joy face_with_tears_of_joy
    thanks brother appreciate effort work put happy_face
    saved aud folded_hands_light_skin_tone subscribed thumbs_up_medium light_skin_tone
    literally binge watching video day quickly becoming favorite channel love see analyze jordan air max vapormaxes future
    see politics talking criminalize fake news make laugh world today democratic capitalism gon na destroi value future kid lie like lie everywhere worst o like theybare caught nobody care o normal problem nike even scaming people
    got every nike adidas yeezy jordan focking shiet surprise obvious everything world shit quality rare find something good
    word word ripped perfectly good shoot freaking american
    would waste perfect shoe people could really need shoe wow ashamed
    airmax please
    please nike adapt auto max self lacing shoe would make day week month tear off_calendar calendar spiral_calendar beaming_face_with_smiling_eyes
    please nike adapt auto max self lacing shoe would make day week month tear off_calendar beaming_face_with_smiling_eyes
    would love nike kyrie shoe nike adapt shoe nike adapt auto max maybe nike kyrie
    would love nike kyrie shoe nike adapt shoe nike adapt auto max maybe nike kyrie believe kyrie leather textile would love see exactly textile even
    loved green screen bro able show important information background explaining helped know actually talking sometimes cause knew nothing leather thanks know alot keep doin ur thing boy thumbs_up never seen video like ur sneaker ur great give much information really test everything subscribing sure saw one ur video proceeded watch another ur ina row hahaha even one like interested shoe ur testing glad stumbled upon ur channel bro literally learned much today presciate bro oncoming_fist edit would love nike kyrie shoe nike adapt shoe
    worn child
    witherspoon air max count
    love air max imo super comfy ok_hand love em
    good sneaker money
    cut air max air max
    would reconmend healthcare worker
    bought pair really nice see shoe actually consists great description
    great content
    nike marketing team atrocious year zero understanding heritage air sac connected chamber exist older model
    call false advertising
    cut yellow yeezy foam sneaker one look like big bird sneaker grinning_squinting_face
    ahh cringed cut shoe wish got pair expensive heck
    always big fan air max shoe either would get new jordan day
    reminds australian aboriginal flag
    want styling video
    inch face_with_tears_of_joy
    came felt like cloud today wear sometimes cool looking memory tech feel outdated heavy bulky
    got euro nike outlet store sale percent bought thel belguim maasmechelen village nike factory
    really good
    add inch height
    got color nike page
    thanks
    vapor max plus decide thinking_face
    wear size vapor max also get air max size
    get half size regular size felt narrow front
    quick video informative
    shoe disgusting honestly scared
    think trip white absolute favourite white nike smiling_face_with_heart eye
    wear squeak
    top rew nice man
    buddy mine bought tried uncomfortable like though nike revolution comfortable
    air force size u men size get air max wide footed okay air force
    actually copped pair last weekend nike back wall though normally wear size still good fit
    ordered
    got mine facebook marketplace gold color way
    get shirt bro
    day buying cheap car cheaper shoe
    got traveling europe bad experience wearing non cushioned shoe cobble stone street perfect got sore able walk hour pair
    always half size unless want want want lose circulation foot
    good sport shoe
    waving_hand question good winter keep warm waterproof
    appreciate review got pair raising_hands_medium_skin_tone
    found beat pair still comfortable face_with_tears_of_joy
    like love way black want also white version tho
    hello question shoe light
    white lemon wash review colour
    let review air max video bro
    confortable
    anyone still comfortable long hour standing walking work retail thinking picking pair
    kiss air foray
    watched found subscribing bros vibe peak
    g one
    man chill honest video
    could someone help cant decide new balance
    review air max
    got first pair yesterday waiting good day wear
    hi comfortable shoe use day day last lifetime secondly air force buy give best additional height
    yes size always crying_face
    maybe someone help im trying buy triple black saw triple black og way cheaper look exactly whats difference two little difference get og one since cheaper
    anybody know foot get super sweaty
    broke really broke like literaly sole broke
    get shirt look clean
    fav sneaker sadly country need work month able buy
    thought something screen
    glow dark nike webside glow im sure
    love get pair man pretty expensive
    im mean never find good shoe right size found air max tho leather fake sad_face
    nike air max good comfortable sport running long walk shoe also good traction
    anyone brown discolouring sponge behind mesh part
    got love em got shoe
    sell em australia eshays face_with_tears_of_joy consider one though bogan
    straight damn point
    size get
    copped black friday sale
    shoe garbage opinion worth paid hard brick lace long confused_face
    decide white white w gum sole
    w video amp review raising_hands_medium dark_skin_tone subscribed
    bruh literally bought watching like said face_with_tears_of_joy
    got mine got discount
    appreciate review contemplating past day hit add bag nike app grinning_face_with_sweat
    every short king watching like eye mouth eye
    shoe suck
    wear size recommend getting half size
    best product way keep clean yr old bought want protect wear happy_face thanks
    got pair triple white morning mind year wanted sks figured would fade get natural contrast
    would buy white gum colorway worth
    tried pair brand new pair felt like something poking toe hurt like hell anyone else
    high quality content amount sub crazy
    expect bro voice
    subscribed vid done still watching got great review cute would love see vid footwear might recommend people high arch
    cap add height mf make look damn near
    actually first video watched subscribed even finished like vibe energy perspective smaller detail regarding example height sole reference arch sole shape towards toe differentiated channel apart others mostly fuck yo energy broskis keep tha good work
    stumbled video thinking getting channel rock video review best
    feeling exactly started buying new pair different airmax every month payday hahaha wanting pair never sure design especially colourways avaliable like actually wear white one popular cause one like paint youtube custom buy like pair one time one pair looking today stadium grey look nice fabric nylon mesh nice square design seen higher end trainer look good size true size go size triax brought back got pair last month found og usa new pair cool still foot light n comfortable compared airmax amd tn tn heavy compared triax
    man sexy look awesome foot really need help regard toe space top cause tn squashed toe wear disappointed_face
    thank god found video order said top part push toe see ya later way im buying air max best toe folded_hands_light_skin_tone oncoming_fist_light_skin_tone thanks dude
    dip front shoe feel good feel stiff feel air cushion end hope go away keep wearing got tho nike
    nice looking triple white comfort concerned beat air max air max though
    love
    seller china selling original nike
    looking pair shoe work white fit look perfect thanks review
    hi great review good summer warm summer
    got honestly none problem probably narrow long foot flatter arch
    need crease protector shoe
    dude training shoe
    wow might get
    white orange version beautiful excited buying sneaker much buying pair go size problem made prolonged walking h walking around city foot burning hot cleaning hard tbh paid around returning take pic memory adorable shoe imo thanks review
    planning buy nd hand airmax watching kinda hesitant last upc digit shoe fake
    jordan zion teardown
    would rate confort wise somone need mix two world normal everyday comfort still able tighten perform needed find play sport matter always un planned sorta pickup game range tennis baseball soccer flag football basketball although im super tall usually dont even bither embarrassing basketball court hahaha game like horse messing around even dedicated sport wear dont find carrying around planning often enough make truly worthwhile im always struggling find best arounder wich tough younger dude basically wore af weird jordan nice degree strap way could comfortable time felf need nad lace bit tighter sinch strap ready put work world technology changed surely must better option styling also important imagine hard choice hahaha
    bought one hour ago south korea really appreciate honest review really helpful thanks mr zach
    good review
    well think k lace ankle brace
    dick sporting good
    shoe super comfortable pillowy good foot long time
    think would good outdoors
    hi use casual shoe think good idea use harsh surface last long enough
    good daily outdoor use
    honest review always definitely getting deldons raising_hands_medium light_skin_tone
    right nike app
    good pg
    men wear
    wait review either peak taichi flash peak ag pro fire hundred_points
    go size higher lower shoe
    must say got deldons found rly comfy harder get go size found size
    great great review weartesters go immediately buy anything got call guy immediately went foot doctor zach
    softer foot pg
    great video always loved design think playstyle might recommend friend
    sneaker awesome two pair
    nike actually sent free pair shoe signed elena delle donna autographed picture sure pretty cool want buy pair size try though
    converse star bb shift something different really curious made insole info given website
    dame certified
    thank may god bless everyone
    wow video mentioned resemble altras wide toe box altras basically wear day anything work pickleball wow insanely expensive especially sweden high vat think anta shoe would make sense tough buy normal c shaped foot normal width movement heavy playstyle normal weight plantar fasciitis issue thanks awesome video
    would recommend lifestyle usage court use like week office casual running
    young heavy swingman wide footed like go risk unstable shot good
    like nike gave lot latitude design given connection disability illness hope sale well female ballers wnba signature option
    honestly love non sponsored nike shoe value bought zoom gt watching video happened store clerk managed sell new latest hyperdunk hahaha
    great interesting review always btw kid love watching video happily known guy cut apart shoe
    hey zach ur ur favourite shoe play personally
    test asics sky elite ff asics metarise made volleyball think would work basketball
    definitely like look shoe hope come second version stability support
    would define someone ankle roller year twice year
    great video hope cover anta brand basketball shoe let u know one great hooping
    could newest kyrie flytrap
    great review flexed_biceps_dark_skin_tone brain wait review aaron gordon nugget nba ag pro degree fire hundred_points
    hey doc review puma rise nitro
    hey zack could performance review jordan one take
    love depth video since foot dr thought expanding video make suggestion every day shoe people may want suggestion daily walking shoe supportive foot type
    wish quick release love love mid high shoe
    hey zach top basketball shoe far year
    zack review mb
    better cushioning shoe hahaha
    zack kyrie thanks advance
    couple day return plastic nike logo lateral side would create pressure foot making hurt wearing awhile
    look nice comfy good finally see half normal looking nike shoe happy_face
    real question good lb small forward
    really like see cut sneaker face_with_tears_of_joy
    zack review puma stewie
    got black color one day surprised confortable shoe stability feel great bad note slip ice outside job late night leaving one saw even seen movie harold kumar christmas kumar fall yup looked
    got systm black already think worth buy excee better model systm better around
    found ross usd color way white university red saw sale local footlocker well bought needed beater shoe amp thought looked pretty fashionable well turn super comfortable beat death get another pair good value price flat wide foot let tell wear day went fit perfect definitely go half size ya got fat foot like con slippery wet go away break
    bought pair week ago bought new pair kick incredibly comfy value money
    bought euro romania pretty cool pair nike
    work running
    make sound walking smooth surface area
    wht mean systm fake crying_face
    glass woa
    love great review
    one would prefer systm flyknit racer eur
    curious v new huarache runner comfortable walking day work mean half price black grey huarache
    flat foot bit wider consider getting half size eu
    got black job requires standing shift comfortable narrow may wan na size half size usual nike size
    hello plastic side hurt inner arch wear mine arch hurt cause plastic say air max high im sure id return theyll comfortable eventually hope help
    difference air max sc v systm v excee
    shoe super narrow white one came stain replace size color overall good purchase wider foot walking foam crumpled two day wear
    nice review let make nike react basis comfortable systm tried airmax comfortable cant feel air sole
    bought black red white comfortable
    compare nike air max ap
    comfort wise would say comfortable sc systm
    found ross store texas white one yellow nike check decided buy price never heard systm weigh lb bad boy comfortable fit perfect went regular size highly recommend
    comfy apparently narrow
    ordered one pale ivory color way super thrilled delivered debating whether get one settled color way look better design also picked zoom pegasus bought legacy mule mom
    air sc
    ordered green white black comfortable order nd pair blk white
    thanks sir
    one problem think fake shoe look exactly color different black white wolf grey look like
    white green colorways bought rd party less retail comfortable day long heel unusually prominent need mind
    got cream white nude one wait
    please recommend better air max systm sc thanks
    would prefer air max systm invincible
    hello buy size airforce bigger size
    paid air max motif air max sc tbqh prefer cheaper pair worried f ing worth
    air max sc jokingly call air max second choice said extremely light comfortable ordered nike something cheap got white platinum light colour spring coming good hear narrow thin foot love air max ideal people thin foot friend even get ninety unit though glad full rubber sole unlike sc pair said foam lasting quite well tbh
    running shoe
    according material sticker shoe made real leather nice see nike skimping leather quality
    buy nike air max ap nike air max systm one comfy work standing hour relly need something comfortable wear
    swoosh reminds pair nike air trainer ii
    coming back seen green colourway cream sole bro ski soo fucking nice like straight hey
    ordered discounted another colorway shiny plastic bit white something shape shiny bit heel counter quite like wide foot still got usual size nike finger crossed fit well glad know super comfy might walking shoe upcoming travel
    euro india right offer color black red nike logo guess totally worth
    jogging
    price point way aud nike must kidding there nothing great mid sole look terrible buy sc excees complete pas look tacky
    bought son physical education used yet seems sole flexible seems hard think
    airmax like shape look futuristic airmax og colourway
    buy one nike air max system dm bus ware have prize buy two kid hahaha bus buy tree total tree
    spot review narrow foot one aspect mention noticeable length bit long side air max uk far longer wide foot tends abnormal crease right inside loopy wavy design turned ocd bit got annoyed may return nice design though
    thank video hesitating purchasing son purchased watching video
    there fourth colour whoch opinion best grey
    new sub waving_hand fire video broski think dark smoke grey flat pewter light iron ore white color way
    loved review
    nike offer og orange box sneaker cost ugly brown boring box im looking scorpion eye
    bro recently baught want check original please help identify think bit heavy
    sneaker color white black volt pure platinum size paid germany month ago eur yes totally agree sneaker super comfy love sneaker feel like eur sneaker
    broski price go wrong comfort count defo take look one hopefully come different color
    would snowman become glass water glass come
    proper chunkified eh look bad tho clapping_hands clapping_hands ok_hand
    sport direct finest pair right
    since skeptas fell apart looking good winter beater might good shout eye think hold good old british weather face_with_tears_of_joy
    upper giving air max zero vibe probably comfy nike ever owned
    great review look chunky prefer classic air max profile tbh
    word
    good white sneaker get expensive looking something different airforce
    review adidas retropy p pls
    yo broski whats best nike tn think
    detail unit testing something trying work wanted automate ticket moved column depending outcome testing struggling link jira board docker could basic test set video appreciate thanks
    step step course udemy zero hero want learn ai agent
    ai generated image impressive realism creativity next level tool like smythos enable orchestrating multiple ai agent creative task
    smythos highlight ai rapid advancement reshaping world prompting u balance innovation ethical consideration ensure technology benefit society
    ai agent revolutionize fashion industry smythos facilitating take look discover impossibility
    concept collaborative ai agent optimizing workflow smythos intriguing code interface make accessible non tech user could revolutionize content creation youtube
    awesome would possible create high level tutorial
    currently basic agent none seem extremely valuable company hopefully year two better use case production wise come although strongly feel llm funxtion calling integrated removing need agent company waste infra setting agent framework
    provide resource database agent using framework
    possible wordpress theme design wondering make happen save hundred hour designed theme wordpress user
    mind blowing usecases mate
    introduction overview html generation agent test generating agency mobiik analytics agent payment processing company sheet agent esm marketing agency sheet agent e commerce brand insight
    possible fine tune llm positively learning conversation ai agent team performed well project negatively learning conversation agent team performed poorly
    ok implement making mad money angry_face
    tese absolutely beginner level agent build complex agent say page instruction talk functionality different aspect different operational process want agent repeatedly constantly follow instruction smoothly complete task successful outcome
    great see agent work
    simply amazing thanks sharing folded_hands
    great value super exciting follow along development ai agent
    thanks video use gradio instead streamlit curious
    dear arsen could please share detail managed convert figma html link agent editing thanks lot
    great work demonstrating foundational capability others build
    demo crazy trying build something similar ask roughly
    hey idea get yest case using picture website
    puh agent really lame
    awesome project smiling_face_with_heart eye news plan regarding course smiling_face_with_smiling_eyes
    exciting project taking note amp subscribed
    using gradio mockups find useful streamlit
    super transparent valuable video thanks happy_face
    make agent iterate task feedback
    thanks vrsen grateful inspiration keep
    unit test agent awesome wow moment moved task create unit test column reviewing star struck
    great
    price around way thin body also support w wireless charging crazy phone forget subscribe channel case miss full review video winking_face
    seriously get crap
    break easy chip old nm nm apple others toy tablet movie
    love huawei phone huawei mate pro using love
    regardless china desire different west earnest future line design practical amp durable failure long run
    huawei trifold phone apple trihole phone face_with_tears_of_joy face_with_tears_of_joy face_with_tears_of_joy face_with_tears_of_joy
    huawei inch new phone released fold time
    honestly speaking foldable phone practical break become faulty easily time
    double fold double fold double fold double fold double fold mate x one fold xt double fold call tech guy call phone trifold phone stupid except one channel
    applaud huawei daring step cool unique luxurious product however everything flexible concerned durability
    bot
    seen pretty nice looking folding phone one top look nice thing kinda spoil gold color favorite computer pocket genius
    comment hateful west working together hahaha western designer initially pioneered foldable technology always produced china get along bit
    worst phone ever break easily
    paid ccp chinese schill face_with_tears_of_joy generation least behind current phone call tri fold obvious fold twice overpriced brick us tofu electronics poorly replicated stolen tech dog pile_of_poo clown_face
    giant hunk crap wonder give away china nobody outside china want one harmony o make useless dont forget brand new phone already mass screen failure face_with_tears_of_joy
    processor using
    want buy
    ultra thin mm casing mm thin battery
    nobody world could marvelous design idea tri fold item
    sent phone try
    use one screen two screen phone distribute stored energy since battery inside screen
    huawei endorsed u government quality innovation assured
    wow want one amaging phone
    one day huawei make hamburger coke america complain america security seriously threatened face_with_tears_of_joy face_with_tears_of_joy face_with_tears_of_joy
    thought bit toyish last week seen person week friend want one minute pulled everyone around went wow damn stole thunder
    amazing innovative meanwhile apple added stpid button iphone hahaha u lagging behind china science technology engineering especially ev car smartphones drone consumer electronics gadget ai since compete u solution ban
    rubbish wastebasket
    national security face_with_tears_of_joy
    far ahead great job huawei
    sexy sitting pose red_heart
    nobody compete technology drove yangwang car day best car ever driven
    shame u regime preventing westerner enjoying superior product
    buy one become chinese spy
    need buy love
    happens accidentally fold wrong way
    pretty obvious china winning certain techrace even get thrown obstacle possible
    miss huawei car ground floor
    wow interesting huawei good job guy bravo clapping_hands
    even expensive iphone combined
    might explode
    best phone world
    type apps run use familiar like apple google apps uckszuwhgymbdv ujg dfgfylandmmqipuia
    notice nobody making fun made china anymore people suddenly find able afford made china dream face_with_tears_of_joy
    chinese left west dust matter much propaganda western nation told contrary long time china rear view mirror militarian innovation also russia rear view also younger doubt made plan emigrate east asap victory_hand china amp russia
    screen mah battery charge every hour rolling_on_the_floor_laughing
    huawei amazing sanctioned united state slandered south korea even ridiculed india huawei definitely successful
    pro operation wise flawlessly innovative con sticking ugly little black dot front facing camera put screen back dumb selfies shi wan na watch video without look like crack screen
    phone flex
    tech overwhelming maket new product choosing one many nerve crackjng
    apple technologically compared
    korean accused huawei stealing tech even
    let hope wont break month
    really careful since cant use case delicate regular tablet mean playing game least play bc press hard letting kid borrow
    bring back google surely buy
    everyone complains apple selling expensive phone huawei sell phone one even complains price kind retard rather buy iphone pro plus ipad oled still money left
    yeah kirin great compared like snapdragon tho
    foldable phone black color look classic red color look amazing
    great introductory review guess mean storage rom
    piece junk
    first time actually want foldable phone pushing limit look like samsung apple thing huawei something none others done
    hint skepticism video durability exposed plastic screen supposed take seriously
    actually better anything apple show
    iphone cry sure face_with_tears_of_joy face_with_tears_of_joy face_with_tears_of_joy face_with_tears_of_joy
    huawei always god level hardware insanely impressive
    fuck ccp
    battery min empt face_with_tears_of_joy
    damn beautiful
    banning huawei security info new development needed ban company image even leaked phone would ended apple reign huawei still
    apple truly ashamed going copy everything release year later hoping forgot huawei first
    nice phone still see fold line like foldable phone
    chinese buy iphones samsung buy huawei face_with_tears_of_joy face_with_tears_of_joy face_with_tears_of_joy face_with_tears_of_joy face_with_tears_of_joy look reality
    made china
    consumer happy product huawei moving u forward american happy someone reteaching apple true innovation died steve job unfortunately hope fanboys start demanding stagnant shareholder myopia trance apple right
    apple guy got new colour
    shopping online realdevice going awesome
    hinge problem waste money
    watching samsung ultra face_with_tears_of_joy next upgrade fold phone need good battery life x zoom
    f apple b marketing
    imagine u sanction huawei would mile ahead today
    huge gap material appearance actual truth advanced backwardness mean closer god advanced separated god alienated backward karl marx desire revenge god abandoning marxism leninism restoring orthodox chinese devine culture way rectify source
    


```python
# finally, we can visualize our Corpus, which is the collection of all the tokens from all the dataset, after the pre-processing step

corpus = pd.DataFrame(
    np.concatenate(df['text_filtered'].values),
    columns=['word']
)

# the 50 most frequent tokens
corpus.groupby('word').size().sort_values(ascending=False).head(50)
```




    word
    shoe                      218
    air                       147
    like                      110
    nike                      104
    one                        98
    max                        98
    pair                       95
    look                       80
    got                        70
    foot                       66
    comfortable                63
    get                        61
    love                       61
    size                       60
    good                       59
    would                      54
    review                     53
    video                      50
    white                      49
    wear                       48
    great                      47
    day                        47
    face_with_tears_of_joy     46
    buy                        46
    really                     44
    bought                     41
    think                      41
    black                      39
    want                       36
    feel                       34
    even                       33
    made                       32
    make                       31
    see                        31
    sneaker                    31
    time                       31
    new                        30
    year                       30
    still                      29
    way                        29
    better                     28
    much                       28
    thanks                     27
    bubble                     26
    phone                      25
    color                      25
    need                       25
    walking                    24
    im                         24
    nice                       24
    dtype: int64




```python
# if we'd like to inspect which comment contains a specific token (for example "ai"), we can run the following line of code:

df[df['text_joined'].str.contains(' ai ')]['text_display']
```




    820    do you have step by step course in udemy from ...
    821    These AI-generated images are impressive! The ...
    822    SmythOS highlights that AI’s rapid advancement...
    825    The concept of collaborative AI agents optimiz...
    833    Is it possible to fine-tune LLM by positively ...
    838    Great value, super exciting to follow along wi...
    883    Very amazing and innovative!   Meanwhile Apple...
    Name: text_display, dtype: object



### Wordclouds
Wordclouds are a fantastic way to display which words are most common in a given corpus, and they also work to express the big picture or a common opinion on a topic.


```python
vocabulary_pos = pd.DataFrame(np.concatenate(df['text_filtered'].values), columns=['words'])

# Generate wordcloud
wordcloud = (WordCloud(width = 3000,
                      height = 2000,
                      random_state=42,
                      background_color='white',
                      colormap='Set2',
                      collocations=False)
            .generate_from_frequencies(
                vocabulary_pos.groupby('words').size().sort_values(ascending=False).to_dict()
              )
            )
# Plot
plt.figure(figsize=(40, 30))
plt.imshow(wordcloud) 
plt.axis("off");
```


    
![png](Youtube%20Sentiment%20Analysis_files/Youtube%20Sentiment%20Analysis_50_0.png)
    


# Classifying Sentiment<a class="anchor" id="fourth-bullet"></a>

In this step generally we could train a traditional Machine Learning model to tag each comment as "positive", "negative" or "neutral" based on their sentiment, but we don't have a previously treated dataset and most importantly we don't have a labeled target (needed in order to train a supervised model). One of the biggest advantages of LLMs such as OpenAI's ChatGPT is the ability to understand and classify text without a training dataset. That's why I opted to apply a LLM directly to classify the comments.


```python
# function to classify comment sentiments and create a summarization for each video
def analyze_comments(search_query, video_title, comments, api_key):
    # prompts
    prompt_sistema = f"""You are a marketing agent specialized in determining whether a video comments' content has a POSITIVE,
    a NEUTRAL or a NEGATIVE sentiment regarding the subject "{search_query}". After reading all comments, you shall return one
    of the three sentiments: ["POSITIVE", "NEGATIVE", "NEUTRAL"]. Additionally, summarize (max. 5 rows) the most frequent 
    positive and negative aspects and hightlighted features about the subject if applicable and if there are enough comments
    to reinforce those opinions (more than 2 comments). As for the output format to compose your answer, write only the sentiment
    word, a colon, and then the summary text.    
    """
    
    prompt_user=f"Video Title: {video_title}\n"
    for comment in comments:
        prompt_user += f"- {comment}\n"
        
    #print(prompt_sistema)
    #print(prompt_user)

    # initialize client
    cliente = OpenAI(api_key=api_key)
    
    # build response
    response = cliente.chat.completions.create(
        messages = [
            {
                "role":"system",
                "content":prompt_sistema
            },
            {
                "role":"user",
                "content":prompt_user
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=4096
    )

    result = response.choices[0].message.content
    return result
```


```python
for video_id in df['video_id'].unique():
    video_title = df[df['video_id']==video_id]['video_title'].values[0]
    comments = df[df['video_id']==video_id]['text_display'].tolist()
    result = analyze_comments(search_query, video_title, comments, openai_key)
    df_videos.loc[df_videos['video_id'] == video_id, 'sentiment'] = result.split(':', 1)[0].strip()
    df_videos.loc[df_videos['video_id'] == video_id, 'summary'] = result.split(':', 1)[1].strip()
    
df_videos
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>video_title</th>
      <th>video_description</th>
      <th>video_url</th>
      <th>relevance</th>
      <th>sentiment</th>
      <th>summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8y5-UvxoD2E</td>
      <td>You’re wrong about Nike Air Max 270</td>
      <td>If you're ever injured in an accident, you can...</td>
      <td>https://www.youtube.com/watch?v=8y5-UvxoD2E</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>The majority of comments express dissatisfacti...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5JzIrzXV2vI</td>
      <td>Pros &amp;amp; Cons: 2024 Nike Air Max DN Review!</td>
      <td>Shop Hibbett City Gear Here! https://bit.ly/37...</td>
      <td>https://www.youtube.com/watch?v=5JzIrzXV2vI</td>
      <td>RELEVANT</td>
      <td>NEUTRAL</td>
      <td>The comments are mainly mixed, expressing both...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>p_-2u8x4Re0</td>
      <td>Nike Alphafly Next % 2 Shoe Review</td>
      <td></td>
      <td>https://www.youtube.com/watch?v=p_-2u8x4Re0</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>Price is highlighted as a negative aspect, wit...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P2YQHLY8TGw</td>
      <td>Nike Air Max 270 Review Black and White</td>
      <td>The Nike Air Max 270 was originally released i...</td>
      <td>https://www.youtube.com/watch?v=P2YQHLY8TGw</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>The most frequent negative aspect mentioned in...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xGM6g7HTInU</td>
      <td>Nike Air More Uptempo Low</td>
      <td>Shop POIZON here!! #ad #poizon Use my code [TE...</td>
      <td>https://www.youtube.com/watch?v=xGM6g7HTInU</td>
      <td>RELEVANT</td>
      <td>NEUTRAL</td>
      <td>There are mixed opinions on the Nike Air More ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xxrXtdVyn34</td>
      <td>Nike Doesn&amp;#39;t Know What&amp;#39;s Inside Their ...</td>
      <td>Buy some Rose anvil leather goods that EVERYBO...</td>
      <td>https://www.youtube.com/watch?v=xxrXtdVyn34</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>Some viewers expressed dissatisfaction with th...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4iv34wHa8Jk</td>
      <td>Nike Air Max 97 White Review: Not What I Expec...</td>
      <td>Description In this video, we review the Air M...</td>
      <td>https://www.youtube.com/watch?v=4iv34wHa8Jk</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>The most frequent negative aspect mentioned in...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>K-HoJUGc4XM</td>
      <td>Nike Air Deldon Biggest Pros And Cons ( Perfor...</td>
      <td>Grab a pair at Nike: https://geni.us/deldon FR...</td>
      <td>https://www.youtube.com/watch?v=K-HoJUGc4XM</td>
      <td>RELEVANT</td>
      <td>NEUTRAL</td>
      <td>The comments include a mix of questions, thank...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>X1ayLlbBtrg</td>
      <td>Nike Jordan 3 Retro Cement Grey | Unboxing &amp;am...</td>
      <td></td>
      <td>https://www.youtube.com/watch?v=X1ayLlbBtrg</td>
      <td>RELEVANT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>wKEF7-LNhSo</td>
      <td>BEST AIR MAX UNDER £100!? Nike Air Max &amp;quot;S...</td>
      <td>Nike Air Max SYSTM DM9537-001 https://tidd.ly/...</td>
      <td>https://www.youtube.com/watch?v=wKEF7-LNhSo</td>
      <td>RELEVANT</td>
      <td>NEUTRAL\n\nPositive Aspects</td>
      <td>1. Comfortable for all-day wear\n2. Good value...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>hb0j9Qn-KjM</td>
      <td>5 Craziest AI Agents We've Ever Built</td>
      <td>🚀 Uncover the 5 Craziest AI Agents We've Ever ...</td>
      <td>https://www.youtube.com/watch?v=hb0j9Qn-KjM</td>
      <td>NaN</td>
      <td>NEUTRAL</td>
      <td>This video has a mix of comments asking for tu...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M-XfdqPBAMU</td>
      <td>HUAWEI Mate XT Hands-on &amp; Quick Review: Huawei...</td>
      <td>That's my dream foldable phone right now.\n#hu...</td>
      <td>https://www.youtube.com/watch?v=M-XfdqPBAMU</td>
      <td>NaN</td>
      <td>NEGATIVE</td>
      <td>There is skepticism about the durability and p...</td>
    </tr>
  </tbody>
</table>
</div>



At last, we can generate a final summary based on the individual summary texts from each video. The function can be written like such:


```python
def generate_final_summary(search_query, sentiment_list, summary_list, api_key):
    prompt_sistema = f"""You are a marketing agent specialized in summarizing comments and reviews about products and topics.
    You will be given a list of multiple sentiment words (being POSITIVE, NEGATIVE or NEUTRAL), and a list of multiple text 
    excerpts, regarding the regarding the topic "{search_query}". Please provide an overall sentiment, returning only one
    of the three sentiments: ["POSITIVE", "NEGATIVE", "NEUTRAL"], and a single summary (max. 10 rows) based on all the
    provided summaries. In this summary, despite the predominant sentiment, summarize the positive aspects and the negative ones. 
    As for the output format to compose you answer, write: "OVERALL SENTIMENT: ", followed by the predominant sentiment word, 
    then the summary text with the positive and negative aspects.
    """
    prompt_user=f"Sentiment list: {sentiment_list}\nSummary list: {summary_list}"
   
    # initialize client
    cliente = OpenAI(api_key=api_key)
    
    # build response
    response = cliente.chat.completions.create(
        messages = [
            {
                "role":"system",
                "content":prompt_sistema
            },
            {
                "role":"user",
                "content":prompt_user
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=4096
    )

    result = response.choices[0].message.content
    return result
```


```python
result = generate_final_summary(search_query, df_videos.sentiment.tolist(), df_videos.summary.tolist(), openai_key)
```


```python
print(result)
```

    OVERALL SENTIMENT: NEGATIVE
    
    Summary:
    The majority of comments regarding the Nike Air review express dissatisfaction with the comfort and durability of the shoes. Users highlight issues such as poor durability, discomfort, tight fit, heel drop design, and lack of cushioning. Price is also a negative aspect, with comments mentioning the inability to afford the shoes and labeling them as a rip off. Some positive aspects include appreciation for the style and looks of the Nike Air shoes, as well as the relief from foot pain they provide for some users. However, the overall sentiment remains negative due to the prevalent criticism of the comfort and durability issues.
    

### Final tests
Testing with another search topic:


```python
# search for a product
search_query = "shadows of the erdtree review"
df_videos = search_videos(search_query, api_key)

# evaluate relevance
df_videos = classify_video(search_query, df_videos, openai_key)

# add new URLs:
#df_videos = add_videos("https://www.youtube.com/watch?v=hb0j9Qn-KjM", df_videos, api_key)

# get all comments
df_videos_comments = get_video_comments(df_videos)

# applying language detection to each comment
df_videos_comments['language'] = df_videos_comments['text_display'].apply(detect_language)
df = df_videos_comments[df_videos_comments['language'] == 'en']

# pre-processing
stopwords.extend(['href', 'quot', 'br', 'u', 'r', 'lt', 'b'])
stopwords = list(set([unidecode(word) for word in stopwords]))
df['text_filtered'] = df['text_display'].apply(lambda x: preprocessing(x))
df['text_joined'] = df['text_filtered'].apply(lambda x: ' '.join(x))

vocabulary_pos = pd.DataFrame(np.concatenate(df['text_filtered'].values), columns=['words'])

# Generate wordcloud
wordcloud = (WordCloud(width = 3000,
                      height = 2000,
                      random_state=42,
                      background_color='white',
                      colormap='Set2',
                      collocations=False)
            .generate_from_frequencies(
                vocabulary_pos.groupby('words').size().sort_values(ascending=False).to_dict()
              )
            )

# apply sentiment analysis and summarize comments
for video_id in df['video_id'].unique():
    video_title = df[df['video_id']==video_id]['video_title'].values[0]
    comments = df[df['video_id']==video_id]['text_display'].tolist()
    result = analyze_comments(search_query, video_title, comments, openai_key)
    df_videos.loc[df_videos['video_id'] == video_id, 'sentiment'] = result.split(':', 1)[0].strip()
    df_videos.loc[df_videos['video_id'] == video_id, 'summary'] = result.split(':', 1)[1].strip()

# final results
result = generate_final_summary(search_query, df_videos.sentiment.tolist(), df_videos.summary.tolist(), openai_key)
print(result)

# Plot Wordcloud
plt.figure(figsize=(40, 30))
plt.imshow(wordcloud) 
plt.axis("off");
```

    C:\Users\VT418741\AppData\Local\Temp\ipykernel_28888\1959997261.py:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['text_filtered'] = df['text_display'].apply(lambda x: preprocessing(x))
    C:\Users\VT418741\AppData\Local\Temp\ipykernel_28888\1959997261.py:22: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['text_joined'] = df['text_filtered'].apply(lambda x: ' '.join(x))
    

    OVERALL SENTIMENT: NEGATIVE
    Summary: 
    Positive Aspects:
    - Some players praise the beautiful open world, Zelda-esque elements, and amount of content provided for the price.
    - Players appreciate the challenge and unique gameplay experience offered by FromSoftware games.
    - New weapons introduced in the DLC are fun to use with unique features like guard counters.
    - The DLC is highly praised for its difficulty level, verticality, world design, bosses, and NPCs.
    - Some viewers mention enjoying the new weapon types, the interconnected map reminiscent of Dark Souls 1, and the cyberpunk vibe of the game.
    
    Negative Aspects:
    - Criticisms include the DLC feeling rushed in terms of content, lack of enemies and loot in certain areas, and high difficulty.
    - Disappointment with lack of replayability, recycled content, and empty areas in the DLC.
    - Complaints about the repetitive gameplay mechanics, lore inconsistencies, disappointing final boss, and the necessity to find seeds to power up in the game.
    - Issues mentioned include empty world design, frustrating mechanics, unbalanced bosses, and lack of depth in the storyline.
    - Frustrations expressed about the difficulty, lack of quest log, excessive deaths, feeling lost, and pricing of the DLC.
    


    
![png](Youtube%20Sentiment%20Analysis_files/Youtube%20Sentiment%20Analysis_60_2.png)
    



```python
print(df_videos['sentiment'].value_counts())
display(df_videos)
```

    sentiment
    NEGATIVE    8
    POSITIVE    2
    Name: count, dtype: int64
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>video_title</th>
      <th>video_description</th>
      <th>video_url</th>
      <th>relevance</th>
      <th>sentiment</th>
      <th>summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>v2tj-b3DfmI</td>
      <td>Elden Ring: Shadow of the Erdtree DLC Review</td>
      <td>Elden Ring: Shadow of the Erdtree DLC reviewed...</td>
      <td>https://www.youtube.com/watch?v=v2tj-b3DfmI</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>Some users criticize the game for having clunk...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>zXQ5nDrspvo</td>
      <td>Elden Ring: Shadow Of The Erdtree Blew My Mind...</td>
      <td>The best thing I can say on Shadow Of The Erdt...</td>
      <td>https://www.youtube.com/watch?v=zXQ5nDrspvo</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>Some users expressed disappointment with the D...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46B1IVjvEks</td>
      <td>Elden Ring Shadow of the Erdtree - Rapid Fire ...</td>
      <td>Patreon ▻ https://www.patreon.com/AngryJoeShow...</td>
      <td>https://www.youtube.com/watch?v=46B1IVjvEks</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>The commenters express frustration about the d...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JCysFRlTN20</td>
      <td>Shadow of the Erdtree Review - Like a Long Los...</td>
      <td>Thanks to Bandai Namco for a review code for S...</td>
      <td>https://www.youtube.com/watch?v=JCysFRlTN20</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>Some players expressed disappointment with the...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0sOUwHFXad4</td>
      <td>ELDEN RING: Shadow of the Erdtree DLC Review -...</td>
      <td>Back in 2022, FromSoftware revolutionized open...</td>
      <td>https://www.youtube.com/watch?v=0sOUwHFXad4</td>
      <td>RELEVANT</td>
      <td>POSITIVE</td>
      <td>The DLC is described as "ABSOLUTELY STUNNING" ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>--eCdoJdTOg</td>
      <td>Elden Ring: Shadow Of The Erdtree Review - No,...</td>
      <td>Thank you for watching, please subscribe if yo...</td>
      <td>https://www.youtube.com/watch?v=--eCdoJdTOg</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>The comments express disappointment with the l...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9FSJvdMy5Tg</td>
      <td>Shadow Of The Erdtree IS PEAK: 100 HOUR REVIEW</td>
      <td>THE MOST SINCERE AND SPECIAL THANKS TO BANDAI ...</td>
      <td>https://www.youtube.com/watch?v=9FSJvdMy5Tg</td>
      <td>RELEVANT</td>
      <td>POSITIVE</td>
      <td>The DLC is praised for its world design, verti...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>zd74u4lIlfk</td>
      <td>Elden Ring: Shadow of the Erdtree PS5 Review -...</td>
      <td>FromSoftware's biggest ever expansion, Shadow ...</td>
      <td>https://www.youtube.com/watch?v=zd74u4lIlfk</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>The reviewer is criticized for not showcasing ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8x2Vb8_BfGs</td>
      <td>My Brutally HONEST Review for Shadow of the Er...</td>
      <td>Like and subscribe if you enjoyed it! Follow m...</td>
      <td>https://www.youtube.com/watch?v=8x2Vb8_BfGs</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>The majority of comments express disappointmen...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>InhfVKgYwdc</td>
      <td>Elden Ring Shadow of the Erdtree - Before You Buy</td>
      <td>Elden Ring returns with the massive DLC expans...</td>
      <td>https://www.youtube.com/watch?v=InhfVKgYwdc</td>
      <td>RELEVANT</td>
      <td>NEGATIVE</td>
      <td>The most frequent negative aspects mentioned a...</td>
    </tr>
  </tbody>
</table>
</div>



```python
# checking all comments for a specific video

video_id='v2tj-b3DfmI'
video_title = df[df['video_id']==video_id].video_title.values[0]

print(video_title)
print()
for index, row in df[df['video_id']==video_id].iterrows():
    comment = row['text_display']
    print(comment)
    print()
```

    Elden Ring: Shadow of the Erdtree DLC Review
    
    The comments always look different depending on what Game the “fans” decide to D-Ride
    
    Nice. Dlc. <br><br>Time for Demons Souls 2 <br><br>Add flying <br>Beautiful open world <br>Zelda ish <br><br>Another game of the year for from software <br><br>Make it happen.
    
    Big DLC and you totally get your money’s worth for it, but a 9 not a 10 for me. Bloodborne’s The Old Hunters is still king of all DLCs IMO. The time travel, weapons, and unbelievable bosses in that one prove that quality wins over quantity any day.
    
    24902 Margaret Junction
    
    252 Rice Islands
    
    Me who still doesn&#39;t have the dlc and the whole game because of financial trouble 😢
    
    Should you get the original one or the DLC?
    
    I&#39;ve never played elden ring. What are some tips? Starting tonight
    
    It’s great that elder ring is finally over it’s time for get excited for the next game
    
    This dlc is so big it could be its own game
    
    Goty 2024 is a dlc
    
    &quot;Never got lost&quot;...definitely not the case for me.  Most of my time has been spent trying to figure out where to go.  Not a knock against the DLC, though.
    
    For $40 and the amount of content and quality you get ... its basically free! Content easily worth  $60 to $70 barely $80 easy ..stretching almost $90 but no more! 😮 and no less for $40 + tax
    
    Chuck Norris plays this game (and all DLCs)
    
    Masterpiece of all masterpiece kneel in front of him! or be dellu and dont accept the true perfection,its your choise😁
    
    Get out of your mom&#39;s or sister&#39;s basements ffs. You adults now. 😂
    
    15 years later and developers are still copying dragon age origins.
    
    Cfb 25 clears easy
    
    Elden ring is a turd wrapped in gold leaf. It has the most clunky, clumsy, outdated controls in a game I’ve ever played. Until the nerds, who enjoy 80’s game mechanics, stop buying this crap, Fromsoftware will keep making it. One day we might get a great  game with modern day controls, a game that could truly be called a masterpiece.
    
    What the hell you spoil so many bloody bosses what the hell first time watching the review after smashing the DLC and thought I’ll see what they said and you spoil so many thing in the game Jesus Christ.
    
    10/10 lmao, IGN ratings are completely meaningless. This is not a 10/10 DLC by any stretch of the imagination
    
    Im giving this dlc a 5/10 for map design and great rune ruining all quest progress otherwise solid lore but so many empty areas the abyss and frenzy was so boring
    
    Have they finally solved the infinite spawn bug?
    
    Why is he pronouncing it &quot;aired tree?&quot;
    
    <a href="https://www.youtube.com/watch?v=v2tj-b3DfmI&amp;t=330">5:30</a> Literally swinging a weapon without having the stat requirements… These are the people they trust to give reviews… 🧐
    
    Are the new areas in the Shadow Realm visually striking? How does the level design compare to the Lands Between?
    
    ER is a 7 at most (open world ruins the pacing, the legacy dungeons are great tho) and the DLC is an 8 at best. IGN too scared to give it what it deserves because of the backlash
    
    IGN added 1+0 together and gave it a ten.
    
    The perfect DLC dont exist.... <br><br>- THE DLC: -
    
    Zero innovation in almost 15 years.
    
    Fort, Night!
    
    It&#39;s a great game but it&#39;s too hard. 4/10
    
    Lol oy 40 bosses for Elden Ring would be mini
    
    Gave up on the base game since they did the nerfs. Never giving them a cent ever. Keep nerfing!
    
    Sounds like an expansion to me….
    
    Only one minor issue for me- no trophies/achievements. Not a deal breaker, just very disappointing.
    
    The special FX artist who work on this game is amazing!!!
    
    It truly is the ULTIMATE dark fantasy open world game....since it&#39;s initial release nothing hasn&#39;t topped it.😊
    
    there are games from 1 to 10, and then there are games Fromsoftware xD
    
    Total rubbish. Games don&#39;t come any worse than this.
    
    Serious question: does ng+ affect the dlc? Because I&#39;ve seen people on ng+1 and are breezing past radahn, I&#39;m on ng+12 and can&#39;t even kill him.
    
    &quot;great variety in areas&quot; shows a tibia mariner
    
    Is it sill clunky lol. You know, a game entirely based on gameplay with clunky gameplay that NOONE acknowledges. Not weird at all
    
    how can a bad graphics like this get 10/10 . this game should get only 7/10.
    
    most overrated game of all time along with zelda
    
    Improving a masterpiece is no small feat
    
    They rated it Elden Ring out of Elden RIng. You&#39;re welcome.
    
    I&#39;m just glad IGN hired someone that can actually play a video game.
    
    This&#39;s why ign review is a joke😅
    
    Such disappointing DLC.<br><br>&quot;Raise the bar.&quot; The bar stayed where it was which wasn&#39;t very high and bold from FS to begin with.<br><br>&quot;But it never crosses the threshold into unfair.&quot;<br><br>Blatant lies, even the base game constantly did that.
    
    25 hours to beat this seems rushed
    
    Spider-Man Miles Morales and Cyberpunk&#39;s Phabtim liberty DLC&#39;s are the greatest! Are you sure this is a great DLC as well???
    
    Ppl are funny, crying its too hard, these people have never played a DS DLC... They are always brutal, they have to be... Elden Ring was by far the easiest souls game so this has probably shocked ppl. Also you shouldn&#39;t forget it needs to take you back to when you first started Elden Ring as well
    
    <a href="https://www.youtube.com/watch?v=v2tj-b3DfmI&amp;t=460">7:40</a>-<a href="https://www.youtube.com/watch?v=v2tj-b3DfmI&amp;t=512">8:32</a> Regarding exploration possibilities in the Shadowlands
    
    Graphics looks better.
    
    The main game was borderlining on impossible and was almost unplayable. This DLC is worse yet! A DLC that is too hard is not even fun. Only a psycho would get enjoyment from dying over &amp; over thousands of times. That is not fun, it&#39;s just pure frustration. Not worth playing. Time to get a REFUND!
    
    10??? For what??? For messed up story telling, unbalanced map and tears bringing sadness???
    
    Couldn&#39;t even give people a special transition or scene when you first enter the DLC zone, just a Black loading screen lol.
<br>
<br>Mindless hype for more of the same (beyond new weapons) to include more vague story telling. 
<br>
<br>If it sells it sells, shrugs in capitalism.
    
    why my bum itchy
    
    boring... fromsoft died at DS2~
    
    Eldin Ring can do no wrong.
    
    This video literally just regurgitates everything that was in the previews.
    
    10/10 And this is the best 1st day purchases from a person who follow a never pre order rule.
    
    Wait. What IGN are still here?
    
    You know its a goat game when its dlc is also 10/10 !
    
    Its not a DLC anymore its basically Elden Ring 2
    
    IGN pretending to be honest ?
    
    Bro this isnt even a review, more like a tutorial with hella spoilers.
    
    Reviewer runs a bleed build.<br>You&#39;re welcome.
    
    Yes of course the new DLC weapons aren&#39;t as powerful as the base game if you use a broken build (dual wield jumping bleed in the video). <br><br>The weapons are balanced as long as you aren&#39;t using a broken build.
    
    Shadow of the Eyrdtree
    
    im listening but not watching the video. didnt want to spoil my gameplay. out for holidays cant wait to play when 😅 get back. but ibaaw the first few seconds and there seem to be a lot of fire!  hhmm
    
    surface pro 10:  what does a certified check look like?
    
    What a time to be alive
    
    Cant wait for the RKG boys to play this....looks great
    
    when you dlc is still bigger then Skyrim
    
    Reviewer is nasty and brainwashed to give 10 to dat even, even game
    
    Waiting for quantum tv review for this dlc
    
    This review can literally start at <a href="https://www.youtube.com/watch?v=v2tj-b3DfmI&amp;t=558">09:18</a> 😅
    
    And here is me who cannot defeat first knight in the game...😅
    
    And people say this is Game Of The Year? LOL Probably the same people saying The Acolyte is brilliant.<br><br>(SPOILERS BELOW)<br><br>Like the base game, the DLC is visually beautiful. Also like the base game, the DLC is highly repetitive and highly boring.<br><br>The only attractive thing about Elden Ring is the mythology, cryptic though it is. The hours and hours of schlepping around doing repetitive tasks to get tiny morsels of storyline is ridiculous. And it&#39;s the storyline and lore to be discovered which makes the DLC attractive. <br><br>What a disappointment.<br><br>Miquella this. Miquella that. Miquella, Miquella,  Miquella. Both in the base game and the DLC, it&#39;s all you ever hear about. And what happens when you meet <b>finally</b> Miquella The Kind? The one Empyrean who apparently appears kind hearted and can see though the folly of Marika and the Erdtree and promises to make the world a better place? That&#39;s right, we have to lay waste to everything. Why? Because a little voice said so when we got high on poison. And we&#39;re given zero choice in the matter.<br><br>We see Miquella for the first time after being teased for 8 hours straight only to smoke him instantly with near zero explanation. We barely see the guy, talk to him, or really have our curiosity satisfied. Or make any choices for that matter. It&#39;s just The Elden Lord traipsing through another land to embody The Murder Hobo Express. And after all that mindless killing, what about the ending? Well there isn&#39;t one, so thanks for playing. A complete and utter anticlimax. You don&#39;t make anything better. Or worse. It was just a 8 hour killing spree for nothing.<br><br>The people who made this game are obviously fans of The Force Awakens: a two hour tease-of-a-movie to show us Luke Skywalker in the last 60 seconds and then drop the curtain with no explanation. What a joke.
    
    In my opinion, the Elden Ring is an 8/10, so is the DLC.
    
    


```python
df[['video_title', 'text_display']].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_title</th>
      <th>text_display</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Elden Ring: Shadow of the Erdtree DLC Review</td>
      <td>The comments always look different depending o...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Elden Ring: Shadow of the Erdtree DLC Review</td>
      <td>Nice. Dlc. &lt;br&gt;&lt;br&gt;Time for Demons Souls 2 &lt;br...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Elden Ring: Shadow of the Erdtree DLC Review</td>
      <td>Big DLC and you totally get your money’s worth...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# checking sample comments from all videos
for index, row in df.sample(n=10).iterrows():
    comment = row['text_display']
    print(comment)
```

    Lenward, I love it you absolutely deserve that box. The mad lad, love to call you big floppa
    The perfect DLC dont exist.... <br><br>- THE DLC: -
    the dlc doesn&#39;t work and I payed 40 f*cking dollars
    <a href="https://www.youtube.com/watch?v=9FSJvdMy5Tg&amp;t=29">0:29</a> if only people seen the beauty of this dlc instead of “i ran straight to rellana and got my shit kicked in because i had no upgrades. From suck”
    Dodge-roll spam, attack, repeat. So bland and boring at this point.
    @fightincowboy time stamp 503 that hammer I got first try at church of consolidation 😅
    Games like this don‘t deserve a rapid fire review but a standard full review…
    why is not dense enought ? how has the dlc an arcane requirement?
    &quot;My next build-defining item could be in that red area down there.&quot;<br><br>Narrator: it wasn&#39;t
    I still say you should play the games that are the Genesis of from software kings field and the Armored Core games since those games over time is what built the reputation from software is known for
    

# Deployment in Production<a class="anchor" id="fifth-bullet"></a>
In order to deploy this solution to production, we choose to create a simple Streamlit application, containing a form to input the search subject, then the amount of videos to be searched, and a button to confirm the operation, which will trigger the code and output the final summary and the wordcloud.

The Streamlit app was deployed using Streamlit Community Cloud.

# Results and Future Improvements<a class="anchor" id="sixth-bullet"></a>
- For its first version the app functions well and is capable of providing very detailed answers, even when using an older model from OpenAI
- Some improvements that could be made include letting the user choose between other models available, adding an option to add or remove videos to the list before they are analysed, and tuning some parameters from the LLM in order to make it more or less creative with the answers.
- A smart language detector could also be implemented, so all videos from the same language would be redirected to different pre-processing pipelines.

---

[Back to top](#top-bullet)
