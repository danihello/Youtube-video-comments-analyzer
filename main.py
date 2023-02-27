import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import func_util as fu
import re

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from googleapiclient.discovery import build
from PIL import Image

st.set_page_config(layout="wide")
pd.options.display.max_colwidth = 300
youtube_api_key = 'AIzaSyAVwyzVKx3Y8NVhROQ0rk4PDlq-4MAGJYc'

matplotlib.use("agg")
_lock = RendererAgg.lock

st.title('Youtube comments sentiment analysis')
st.write("")



with st.form("video_form"):
    rowS_1, rowS_2, rowS_3, rowS_4 = st.columns(
        (1, 0.3, 0.3, 0.3)
    )
    
    with rowS_1:
        video_link = st.text_input(
            "Input your youtube Link for analysis (e.g. https://www.youtube.com/watch?v=Y3gF6Duh_z0&ab_channel=DisneyPlus)"
        )
    with rowS_2:
        comment_order = st.selectbox(label='comment order', options=['relevance', 'time'])
    with rowS_3:
        maxResults = st.selectbox(label='how many comments to fetch?', options=[100, 200, 300, 400 ,500])
    with rowS_4:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("Submit")

if video_link:
    id = fu.video_id(video_link)
    comment_list = fu.getcomments(id, maxResults=maxResults, order=comment_order)
    comments = fu.parse_comments(comment_list)
    comments['text_clean'] = comments['textOriginal'].apply(fu.clean_text)
    comments['language']=comments['text_clean'].apply(fu.get_lang_2)
    n_comments_api = comments.shape[0]
    comments.drop_duplicates(subset='textOriginal', keep='first', inplace=True)
    comments_english = comments.loc[comments['language'] == 'English']
    n_comments_api_en = comments_english.shape[0]
    n_comments_api_nodup = comments.shape[0]

    video_details = fu.get_video_details(id)
    viewCount, likeCount, commentCount = video_details['statistics']['viewCount'],video_details['statistics']['likeCount'],video_details['statistics']['commentCount']
    publish_date = video_details['published_date']
    publish_date = re.sub(r"[tT]", " ", publish_date)
    publish_date = re.sub(r"[zZ]","",publish_date)
    image = video_details['image_url']
    author = video_details['channelid']

    st.subheader('Video analysis for: **{}**'.format(video_details['title']))
    st.image(image, caption=video_details['title'])
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
        (0.1, 1, 0.2, 1, 0.1)
    )
    with row0_1:
        st.subheader('video duration: **{}**'. format(video_details['duration']))
        st.write("")
        st.subheader('video statistics')
        st.write(f'Views: {int(viewCount):,}')
        st.write(f'Likes: {int(likeCount):,}')
        st.write(f'Comments: {int(commentCount):,}')
        st.write(f'Published date: {publish_date}')
        st.write(f'Published by: {author}')

    with row0_2:
        st.subheader("")
        st.write("")
        st.write("")
        st.write("")
        st.subheader("Youtube v3 API statistics")
        st.write(f'Comments pulled for analysis: {n_comments_api}')
        st.write(f'Comments after removing duplicates: {n_comments_api_nodup}')
        st.write(f'Comments in English: {n_comments_api_en}')
        st.write(f'Comments order: {comment_order}')

    
    comments['sentiment_score'] = comments.loc[comments['language'] == 'English', 'text_clean'].apply(
        fu.sentiment_analysis_compound_score)
    comments['sentiment_ind'] = comments.loc[comments['language'] == 'English', 'text_clean'].apply(
       fu.sentiment_analysis)

    # st.subheader('Raw data')
    # st.dataframe(comments)

    has_records = any(comments['text_clean'])

else:
    has_records = False
    #comments = pd.DataFrame()

#has_records = False
st.write("")
row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

with row1_1, _lock:
    st.subheader("Comments by Sentiemnt distribution")
    if has_records:
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        data = comments['sentiment_ind'].value_counts().to_frame()
        data.insert(1, "colors",
                    pd.Series(['#99ff99', '#66b3ff', '#ff9999'], index=['positive', 'neutral', 'negative']))
        fig = Figure(figsize=(4, 4))
        ax = fig.subplots()
        data.plot(y='sentiment_ind', kind="pie", autopct='%1.1f%%', shadow=True, colors=data['colors'], legend=True,
                  ylabel='', labeldistance=None, ax=ax)
        ax.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
        # ax.set_title("Year")
        st.pyplot(fig)
    else:
        st.markdown("This video has no comments or no english comments")

st.write("")
with row1_2, _lock:
    st.subheader("Comments by Language distribution")
    if has_records:
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        data = fu.condense_categories(comments, 'language', 3, 'Other languages')
        fig = Figure(figsize=(4, 4))
        ax = fig.subplots()
        data.plot(kind="pie", autopct='%1.1f%%', shadow=True, colors=colors, legend=True, ylabel='', labeldistance=None,
                  ax=ax)
        ax.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
        # ax.set_title("Year")
        st.pyplot(fig)
    else:
        st.markdown("This video has no comments or no english comments")

st.write("")
row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

if has_records:
    trend_data = comments.copy()
    ndates = trend_data['published_date'].dt.date.nunique()
    if ndates <=60:
        trend_data['date'] = trend_data['published_date'].dt.date
    else:
        trend_data['date'] = trend_data['published_date'].apply(fu.getyear_month)

    trend_data_agg = trend_data.groupby(['date', 'sentiment_ind'])['published_date'].count().unstack().fillna(0)
    trend_data_agg['total'] = trend_data_agg.sum(axis=1)

with row2_1:
    if has_records:
        st.subheader("Comments frequency time trend")
        fig = plt.figure(figsize=(9,9))

        plt.subplot(2,1,1)
        g1 = sns.lineplot(data=trend_data_agg[['total']])
        g1.set(xticklabels=[])
        g1.set(xlabel=None)
        plt.xticks(rotation=45)
        plt.ylabel('comments')

        plt.subplot(2,1,2)
        g2 = sns.lineplot(data=trend_data_agg[['negative','positive']], dashes=False, palette={'positive':'green', 'negative':'red'})
        plt.xticks(rotation=45)
        plt.ylabel('comments')
        every_nth = 2
        for n, label in enumerate(g1.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        for n, label in enumerate(g2.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        st.pyplot(fig)
    else:
        st.markdown("This video has no comments or no english comments")
with row2_2:
    if has_records:
        st.subheader("Comments frequency by days of the week")
        fig = plt.figure(figsize=(9, 9))

        cats = {'Sunday':1, 'Monday':2, 'Tuesday':3, 'Wednesday':4, 'Thursday':5, 'Friday':6, 'Saturday':7}
        trend_data_2 = comments.copy()
        trend_data_2['day_name'] = trend_data_2['published_date'].dt.day_name()
        trend_data_2 = trend_data_2.groupby(['day_name'])['published_date'].count().fillna(0).reset_index()
        trend_data_2['day_name_order'] = trend_data_2['day_name'].map(cats)
        trend_data_2 = trend_data_2.sort_values('day_name_order')
        trend_data_2.rename(columns={'published_date':'comments'}, inplace=True)

        pal = sns.color_palette("Greens_r", len(trend_data_2))
        rank = trend_data_2.set_index('day_name')['comments'].argsort().argsort()
        ax = sns.barplot(data=trend_data_2, x='day_name', y='comments', palette=np.array(pal[::-1])[rank])
        plt.xticks(rotation=45)
        ax.set(xlabel=None)
        st.pyplot(fig)
    else:
        st.markdown("This video has no comments or no english comments")


st.write("")
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

with row3_1:
    st.subheader("WordCloud where sentiment is 'Positive'")
    if has_records:
        data = comments.loc[comments['sentiment_ind'] == 'positive', ['text_clean']].squeeze()
        wc = fu.get_wordcloud(data)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)
    else:
        st.markdown("This video has no comments or no english comments")

with row3_2:
    st.subheader("WordCloud where sentiment is 'Negative'")
    if has_records:
        data = comments.loc[comments['sentiment_ind'] == 'negative', ['text_clean']].squeeze()
        wc = fu.get_wordcloud(data)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)
    else:
        st.markdown("This video has no comments or no english commentd")

#has_records = False

st.write("")
row4_space1, row4_1, row4_space2 = st.columns(
    (0.1, 2.1, 0.1)
)
with row4_1:
    st.subheader("Sentiment breakdown by emotions")
    if has_records:
        input = fu.oneai_input_emotions(comments['text_clean'])
        emotions = fu.oneai_get_emotions(input)

        emotions_df = pd.DataFrame(
            [{key: value for key, value in element.items() if key in ['speaker', 'name', 'span_text']} for element in emotions])
        emotions_df['speaker'] = emotions_df['speaker'].astype('int')
        emotions_agg_df = emotions_df.groupby('speaker')['name'].value_counts().unstack().sort_index()
        comments_emo = comments.join(emotions_agg_df).fillna(0)
        comments_emo[['anger', 'happiness', 'sadness','surprise']] =  comments_emo[['anger', 'happiness', 'sadness','surprise']].applymap(lambda x:1 if x > 0 else 0)
        emotions_data = comments_emo.loc[comments_emo['sentiment_ind']!=0].groupby('sentiment_ind')\
            ['sadness','happiness','anger','surprise'].sum().reset_index().melt(id_vars='sentiment_ind',var_name='emotion',value_name='comments')
        
        sns.set_theme(style="whitegrid")
        sns.set_palette("muted")
        color_palette = {'sadness':'blue', 'happiness':'orange', 'anger':'red', 'surprise':'purple'}
        fig = sns.catplot(data=emotions_data, kind='bar', col='sentiment_ind', x='emotion', y='comments', orient='v', sharey=False, aspect=0.85, palette=color_palette)
        st.pyplot(fig)
    else:
        st.markdown("This video has no comments or no english commentd")


st.write("")
row5_space1, row5_1, row5_space2 = st.columns(
    (0.1, 2.1, 0.1)
)
with row5_1:
    st.subheader("Showing top 10 comments by engagement, likes and replys)")
    if has_records:
        comments_emo['engagement'] = (15*comments_emo['replyCount']+comments_emo['likeCount'])
        st.dataframe(comments_emo.sort_values('engagement', ascending=False).head(10))
    else:
        st.markdown("This video has no comments or no english commentd")