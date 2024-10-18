import pandas as pd
import numpy as np
import nltk
import emoji
import re
import requests
import os
import streamlit as st
import time
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
from dotenv import load_dotenv

# settings
pd.set_option('display.float_format', lambda x: '%.2f' % x)
st.set_page_config(layout="wide")

# import keys
load_dotenv()
api_key = os.getenv("API_KEY")
#openai_key = os.getenv("OPENAI_API_KEY")

# function to load API key and start Youtube API
def load_api(api_key):
    # Set up the API
    youtube = build('youtube', 'v3', developerKey=api_key)
    return youtube

# function to search for videos based on search query
def search_videos(search_query, max_results, api_key):
    try:
        if max_results > 50: # Number of videos to retrieve
            max_results = 50

        API_KEY = api_key

        url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults={max_results}&q={search_query}&type=video&relevanceLanguage=en&key={API_KEY}"

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
    except Exception as err:
        raise Exception(' search_videos:' + str(err))

# function to classify video title as relevant or not
def classify_video(search_query, df_videos, api_key):
    try:
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
    except Exception as err:
        raise Exception(' classify_video:' + str(err))

# function to add video to the candidates list
def add_videos(video_url, df_videos, api_key):
    try:
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
    except Exception as err:
        raise Exception(' add_videos:' + str(err))

# pre-processing
# get all comments from a SINGLE video and save them into a list
def get_comments(video_id):
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=10000
        )

        comments = []
        try:
            response = request.execute()

            # creating a list
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
        except:
            pass

        return comments
    except Exception as err:
        raise Exception(' get_comments:' + str(err))

# obtain comments for ALL the videos (except "NOT RELEVANT" ones) and save into a dataframe
def get_video_comments(df_videos):
    try:
        all_video_details = []

        for index, row in df_videos.iterrows():
            video_id = row['video_id']
            video_relevance = row['relevance']
            
            if video_relevance != "NOT RELEVANT":
                comments = get_comments(video_id)
                if len(comments) != 0:
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
    except Exception as err:
        raise Exception(' get_video_comments:' + str(err))

# obtain stopwords
def get_stopwords():
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords = list(set([unidecode(word) for word in stopwords])) # remove accentuation
    stopwords.extend(['href', 'quot', 'br', 'u', 'r', 'lt', 'b'])
    stopwords = list(set([unidecode(word) for word in stopwords]))

    return stopwords

# Function to detect the language
def detect_language(text):
    DetectorFactory.seed = 42
    try:
        return detect(text)
    except:
        return 'unknown'

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

# pre-processing function - pipeline
def preprocessing(string):
    try:
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
        stopwords = get_stopwords()

        filtered_words = []
        for w in words:
            if w not in stopwords:
                filtered_words.append(w)
        
        # lemmatization
        lemmatizer = WordNetLemmatizer()
        lemma_words = []
        for w in filtered_words:
            l_words = lemmatizer.lemmatize(w)
            lemma_words.append(l_words)
            
        return lemma_words

    except Exception as err:
        raise Exception(' preprocessing:' + str(err))

# plot the wordcloud
def plot_wordcloud(df):
    try:
        vocabulary_pos = pd.DataFrame(np.concatenate(df['text_filtered'].values), columns=['words'])

        # Generate wordcloud
        wordcloud = (WordCloud(width = 2000,
                            height = 1000,
                            random_state=42,
                            background_color='white',
                            colormap='Set2',
                            collocations=False)
                    .generate_from_frequencies(
                        vocabulary_pos.groupby('words').size().sort_values(ascending=False).to_dict()
                    )
                    )
        # plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis("off")
        # plt.show()
        # st.pyplot()

        # Set the figure size
        plt.figure(figsize=(10, 6))
        # Display the word cloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    except Exception as err:
        raise Exception(' plot wordcloud:' + str(err))

# only consider english comments
def filter_english(df):
    df_videos_comments['language'] = df_videos_comments['text_display'].apply(detect_language)
    df = df_videos_comments[df_videos_comments['language'] == 'en']
    return df

# function to classify comment sentiments and create a summarization for each video
def analyze_comments(search_query, video_title, comments, api_key):
    try:
        # prompts
        prompt_sistema = f"""You are a marketing agent specialized in determining whether a video comments' content has a POSITIVE,
        a NEUTRAL or a NEGATIVE sentiment regarding the subject "{search_query}". Do not consider comments about the video itself (quality, audio or uploaded time/format), only consider those about the subject. After reading all comments, if they did not provide any significant opinion, do not return anything. After reaching an opinion, you must return one
        of the four options, whichever fits best what you learned from the comments: ["POSITIVE", "NEGATIVE", "NEUTRAL", "NONE"]. The three first are sentiments and the fourth option, "NONE", is for when no significant opinion were provided through the comments. Additionally, summarize (max. 5 rows) the most frequent positive and negative aspects and hightlighted features about the subject if applicable and if there are enough comments to reinforce those opinions (more than 2 comments). As for the output format to compose your answer, write spefically in this format, and only this alone: one of the four options ["POSITIVE", "NEGATIVE", "NEUTRAL", "NONE"], a colon, and then the summary text. Example of answer: "POSITIVE: good pricing and good impressions on...". You must return some text after the sentiment.
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
    except Exception as err:
        raise Exception(' analyze_comments:' + str(err))

# generate final summary for video list
def generate_final_summary(search_query, sentiment_list, summary_list, api_key):
    try:
        prompt_sistema = f"""You are a marketing agent specialized in summarizing comments and reviews about products and topics.
        You will be given a list of multiple sentiment words (being POSITIVE, NEGATIVE or NEUTRAL), and their respective list of text 
        excerpts, regarding the topic "{search_query}" from various Youtube videos. Please provide an overall sentiment that can be assigned to the majority of the summaries, 
        by calculating the most frequent sentiment word, returning only one of the three following sentiments as the most representative: ["POSITIVE", "NEGATIVE", "NEUTRAL"], 
        and a single summary (max. 10 rows) based on all the provided excerpts. In this summary, write about both the positive aspects as well as the negative ones (no matter which the most predominant sentiment was), including the words "POSITIVE" and "NEGATIVE" as separate topics.
        As for the output format to compose you answer, write: "OVERALL SENTIMENT: ", followed by the predominant sentiment word, 
        then break the line and write the summary text separating the positive and negative aspects.
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
    except Exception as err:
        raise Exception(' generate_final_summary:' + str(err))


# code for testing
if __name__ == '__main__':
    try:
        youtube = load_api(api_key)

        button_disabled = True
        button_2_disabled = True
        # Create a placeholder for updates
        st.title('Youtube Comments Sentiment Analysis')
        st.write('This is a solution aimed at identifying the most predominant sentiment from the comments of Youtube videos that are returned as search results for a given query expression. Once you fill the subject to be searched and the amount of videos to be considered from the search results (Youtube API will search for the most relevant ones), just click on "Search" in order to obtain a custom AI-generated review and summary from all the relevant comments, as well as a wordcloud cointaing the most recurring words within them.')

        openai_key = st.text_input('OpenAI Key')
        st.markdown(":bulb: LLM used: gpt-3.5-turbo")


        st.header( 'Input form' )
        search_query = st.text_input('Please input expression to be searched in Youtube', "Ray-Ban Meta Smart Glasses review")
        max_results = st.number_input('Amount of videos to be searched (min. 5 - max. 50)', value=10)
        try:
            int(max_results)
        except ValueError:
            raise Exception(' invalid amount of video results was inputted.')
        
        if len(search_query) > 3 and max_results >= 5 and len(openai_key) > 10:
            button_disabled = False

        if st.button("Search", disabled=button_disabled):
            st.header( 'Results' )
            with st.empty():
                button_disabled = True
                st.write("Processing, please wait a short while...")

                #search_query = "shadows of the erdtree review"
                st.write("Searching videos...")
                df_videos = search_videos(search_query, max_results, api_key)

                # evaluate relevance
                st.write("Determining videos' relevance...")
                df_videos = classify_video(search_query, df_videos, openai_key)

                # add new URLs:
                #df_videos = add_videos("https://www.youtube.com/watch?v=hb0j9Qn-KjM", df_videos, api_key)

                # get all comments
                st.write("Obtaining all videos' comments...")
                df_videos_comments = get_video_comments(df_videos)

                # applying pre-processing pipeline
                df = filter_english(df_videos_comments)
                df['text_filtered'] = df['text_display'].apply(lambda x: preprocessing(x))
                df['text_joined'] = df['text_filtered'].apply(lambda x: ' '.join(x))

                # apply sentiment analysis and summarize comments
                st.write("Analyzing all comments and preparing summary...")
                for video_id in df['video_id'].unique():
                    try:
                        video_title = df[df['video_id']==video_id]['video_title'].values[0]
                        comments = df[df['video_id']==video_id]['text_display'].tolist()
                        print(video_id)
                        result = analyze_comments(search_query, video_title, comments, openai_key)
                        print(result)
                        if "NONE" in result or ("POSITIVE:" not in result and "NEGATIVE:" not in result and "NEUTRAL:" not in result):
                            pass
                        else:
                            df_videos.loc[df_videos['video_id'] == video_id, 'sentiment'] = result.split(':', 1)[0].strip()
                            df_videos.loc[df_videos['video_id'] == video_id, 'summary'] = result.split(':', 1)[1].strip()
                    except:
                        pass

                # final results
                st.write("Generating final results...")
                result = generate_final_summary(search_query, df_videos.sentiment.tolist(), df_videos.summary.tolist(), openai_key)
                #print(result)

            # plotting wordcloud
            st.write("Plotting wordcloud...")
            plot_wordcloud(df)

            # write final results
            st.write(result)

            st.write('Displaying videos analysed:')
            st.dataframe(df_videos[['video_title', 'video_url', 'sentiment', 'summary']].dropna(subset=['sentiment']))

    except Exception as err:
        st.write( 'Erro: ' + str(err) )


