import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
import seaborn as sns
import isodate

from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from langdetect import detect, detect_langs
from langdetect import DetectorFactory
from iso639 import languages
from polyglot.detect import Detector
from emoji import demojize, replace_emoji

from wordcloud import WordCloud
from wordcloud import ImageColorGenerator

DetectorFactory.seed = 1

api_key = 'AIzaSyAVwyzVKx3Y8NVhROQ0rk4PDlq-4MAGJYc'


import streamlit as st

def video_id(value):
    """
    Parses the video id
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse(value)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    # fail?
    return None

@st.cache
def getcomments(videoId, maxResults=200, order='relevance'):
    """
    calls oyutube v3 API and gets the comments in a dictionary
    Args:
        videoId (str): the videoid for the API to get the comments from
        maxResults(int): number of comments to get from the API (default is 200)
        order (str): the order in which the data gets pulled either by relevane or time (default is felevane)
    
    Returns:
        list: list of dictionaries containing all the comments pulled from the API
    """
    response = []
    
    
    MAX_RESULTS = maxResults
    
    # creating youtube resource object
    youtube = build('youtube', 'v3',
                    developerKey=api_key)
    
    # the max results we can get from the api is 100  we need to make sure not to get more then that
    if maxResults > 100:
        n_results = 100
    else:
        n_results = maxResults
    
    #getting comments using youtube api
    video_response=youtube.commentThreads().list(
            part='snippet',
            order=order,
            maxResults=n_results,
            videoId=videoId,
            ).execute()
    

    #saving the comments in the response variable
    response = video_response['items']
    #getting data for the next page if needed
    ntp = video_response.get('nextPageToken')

    #looping the api until we get to MAX_RESULTS number

    while len(response)<MAX_RESULTS and 'nextPageToken' in video_response:

        maxResults -= len(video_response['items'])
        if maxResults > 100:
            n_results = 100
        else:
            n_results = maxResults
        
        if ntp:
            video_response=youtube.commentThreads().list(
            part='snippet',
            order=order,
            maxResults=n_results,
            pageToken=ntp,
            videoId=videoId,
            ).execute()

        response += video_response['items']
        ntp = video_response.get('nextPageToken')
        
    return response

@st.cache(allow_output_mutation=True)
def parse_comments(comments):
    """
    extracts the necessary data from a list of dictionaries and returns a DataFrame

    Args:
        comments (list of dictionaries) - containing the comments to parse

    Returns:
        DataFrame of commnets with the neded data
    """
    #extracting the relevent data from the json
    comments_dict =[{'textOriginal':comment.get('snippet').get('topLevelComment').get('snippet').get('textOriginal'),
            'likeCount':comment.get('snippet').get('topLevelComment').get('snippet').get('likeCount'),
            'published_date':comment.get('snippet').get('topLevelComment').get('snippet').get('publishedAt'),
            'replyCount':comment.get('snippet').get('totalReplyCount')} 
           for comment in comments]
    #converting the dict to dataframe and converting the string to datetime
    df = pd.DataFrame(comments_dict)
    df['published_date'] = pd.to_datetime(df['published_date'],format='%Y-%m-%dT%H:%M:%SZ')
    return df

@st.cache(allow_output_mutation=True)
def get_language(text):
    """
    uses detect langs to find the language this function was replaced by 'get_lang_2'
    """
    try:
        iso_code = detect_langs(text)[0].lang
        return languages.get(alpha2=iso_code).name
    except:
        return 'no lang'

@st.cache(allow_output_mutation=True)
def get_lang_2(text):
    """
    detects the language given

    Args:
        text (str): a string of text

    Returns:
        str: the language used in the text variable, returns 'un' if no language was found
    """
    return Detector(text, quiet=True).language.name

@st.cache(allow_output_mutation=True)
def sentiment_analysis_compound_score(text):
    """
    returns the compound_score predicted by the 'SentimentIntensityAnalyzer'
    
    Args:
        text (str): a phrase to predict a sentiment scroe

    Returns:
        float: a score between -1 and 1 above 0 is positive and below is negative
    """
    
    sid = SentimentIntensityAnalyzer()    
    return sid.polarity_scores(text).get('compound')

@st.cache(allow_output_mutation=True)
def sentiment_analysis(text, neutral_limit = 0.05):
    """
    returns a sentiment label based on sentiment_analysis_compound score

    Args:
        text (str): a phrase to predict a sentiment label
        neutral_limir (float): a limit that creates a third neutral label

    Returns:
        str: a label of either poritive, negative or neutral based on neural_limit parameter
             and the compound score
    """
    compound = sentiment_analysis_compound_score(text)
    if compound > neutral_limit:
        return 'positive'
    elif compound < -neutral_limit:
        return 'negative'
    else:
        return 'neutral'

# Edit this One AI API call using our studio at https://studio.oneai.com/?pipeline=3aRxk5


def get_wordcloud(series, background_color='black'):
    """
    returns a wordcloud image based on the text given

    Args:
        series (List or np Array or pd Series): contains strings of text

    Returns:
        img: an image of the most frequent words 
    """
    text = " ".join(word for word in series)
    stop_words = stopwords.words("english")
    wordcloud = WordCloud(stopwords=stop_words, background_color=background_color).generate(text)
    return wordcloud
    #plt.figure( figsize=(15,10))
    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()

def clean_text(phrase):
    '''
    do various cleaning tasks
    
    Args:
        phrase (str): String to which the function is to be applied, string
    
    Returns:
        str: clean string of text
    '''
    def decontracted(phrase):
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    phrase = phrase.lower()
    phrase = re.sub(r'(http|@)\S+', "",phrase)
    phrase = replace_emoji(phrase,replace="")
    phrase = re.sub(r"’", "'",phrase)
    phrase = re.sub(r"[^a-z\':_]", " ",phrase)
    phrase = decontracted(phrase)
    phrase = phrase.replace(":","")
    phrase = phrase.replace("'","")
    phrase = re.sub(r'^\s*|\s\s*', ' ', phrase).strip()
    return phrase

def remove_english_stopwords_func(text):
    '''
    Removes Stop Words (also capitalized) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without Stop Words
    ''' 
    # check in lowercase 

    t = [token for token in word_tokenize(text) if token.lower() not in stopwords.words("english")]
    text = ' '.join(t)    
    return text

def condense_categories(df, col_name, max_cat, other_cat_name='other'):
    """
    returns a series of categories based on parameters
    consolidating categories with few occurences into a new category

    Args:
        df (DataFrame): a dataframe containing the information
        col_name (str): name of the column in the dataframe we with to consolidate
        max_cat (int): maximum number of categories we wish
        other_cat_name (str): name of the new category

    Returns:
        Series: returns a frequency series 
                with a new category with the less frequent categories
    """
    df_freq = df[col_name].value_counts().head(max_cat)
    other_count = df[col_name].value_counts()[max_cat:].sum()
    if other_count:
        df_freq.loc[other_cat_name] = other_count
    return df_freq

@st.cache
def get_video_details(video_id):
    """
    returns a dictionary with a youtube video information

    Args:
        video_id (str): a string representing a video id for the youtube API

    Returns:
        dictionary: containing various information about the youtube video
    """
    details_dict={}
    youtube = build('youtube', 'v3',
                        developerKey=api_key)
    video_details=youtube.videos().list(
    part='snippet,statistics,contentDetails',          
    id=video_id,
    ).execute()
    details_dict['title']=video_details['items'][0]['snippet']['title']
    dur = isodate.parse_duration(video_details['items'][0]['contentDetails']['duration'])
    details_dict['duration']=dur
    details_dict['published_date'] = video_details['items'][0]['snippet']['publishedAt']
    details_dict['image_url'] = video_details['items'][0]['snippet']['thumbnails']['high']['url']
    details_dict['channelid'] = video_details['items'][0]['snippet']['channelTitle']
    details_dict['statistics']=video_details['items'][0]['statistics']
    return details_dict

@st.cache(allow_output_mutation=True)
def oneai_input_emotions(series):
    """
    creates a template input for the OneAI API

    Args:
        series (iterable): an iterable containing utterance for the OneAI API

    Returns:
        list: list of dictionaries for the OneAI API see One AI documentation https://docs.oneai.com/docs
    """
    text_dict = series.to_dict()
    input = [{"speaker":str(key), "utterance":value} for key, value in text_dict.items()]
    return input

@st.cache(allow_output_mutation=True)
def oneai_get_emotions(input):
    # Edit this One AI API call using our studio at https://studio.oneai.com/?pipeline=gZ9b3p
    """
    returns a JSON response from the OneAI API with skill 'emotions'

    Args:
        input list: input for the API

    Returns:
        JSON: a dictionary with the response to a request sent using the OneAI API
    """
    api_key = "40b7dbd3-eb8c-4aca-b395-95a80706a6ed"
    url = "https://api.oneai.com/api/v0/pipeline"
    headers = {
    "api-key": api_key, 
    "content-type": "application/json"
    }
    payload = {
    "input": input,
    "input_type": "conversation",
        "content_type": "application/json",
    "steps": [
        {
        "skill": "emotions"
        }
        ],
    }
    r = requests.post(url, json=payload, headers=headers)
    data = r.json()
    try:
        return data['output'][0]['labels']
    except:
        return 'no emotions detected'

def getyear_month(date):
    """
    returns a yyyy-MM string based on input

    Args:
        date (datetime): date input
    
    Returns:
        a string with a format of yyyy-MM based on date given as input
    """
    if date.month<10:
        return str(date.year)+'-'+'0'+str(date.month)
    return     str(date.year)+'-'+str(date.month)