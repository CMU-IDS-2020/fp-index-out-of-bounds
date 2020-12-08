import streamlit as st
import numpy as np
import pandas as pd
import random
import altair as alt
from wordcloud import WordCloud
from textblob import TextBlob
from sentiment_transfer import sentiment_transfer

## model and data storage

sentiment_evaluation_model = None
sentiment_transformation_model = None
style_transfer_model = None
data = None

## two functions for testing

def __generate_random_sentences__(num_sample = 5000):
    nouns = ("puppy", "car", "rabbit", "girl", "monkey")
    verbs = ("runs", "hits", "jumps", "drives", "barfs") 
    adv = ("crazily", "dutifully", "foolishly", "merrily", "occasionally")
    adj = ("adorable", "clueless", "dirty", "odd", "stupid")
    sample_list = []
    for i in range(num_sample):
        sample_list.append([nouns[random.randrange(0,5)] + ' ' + verbs[random.randrange(0,5)] + ' ' + adv[random.randrange(0,5)] + ' ' + adj[random.randrange(0,5)],'Speaker A' if random.randint(0, 1) else 'Speaker B'])
    return pd.DataFrame(sample_list, columns = ['Sentences', 'Speaker'])

def __generate_random_scores__(num_sample = 5000, mean=2.5):
    sentiment_scores = np.random.randn(num_sample)*0.7+mean
    sentiment_scores = pd.DataFrame(data=sentiment_scores, columns=['Sentiment'])
    return sentiment_scores

def __transform_sent_with_model__(sentences, target_level):
    re = []
    for s in sentences:
        re.append(sentiment_transfer(s, target_level))
    return pd.DataFrame(re)


def __transform_styl_with_model__(sentences):
    pass

def __evaluate_sent_with_model__(sentences):
    # clean_data = [clean_line(s) for s in sentences]
    return pd.DataFrame([TextBlob(s).sentiment.polarity for s in sentences]).apply(lambda x: 10*x)

def __sentiment_evaluation__(sentences, test = False, target=None):
    num_of_sentences = len(sentences)
    if sentiment_evaluation_model is None or test:
        if target is not None:
            scores = __generate_random_scores__(num_of_sentences, target)
        else:
            scores = __generate_random_scores__(num_of_sentences)
    else:
        scores = __evaluate_sent_with_model__(sentences)
    # else do inference using the evaluation model
    return scores

## application initialization functions

def get_sentiment_evaluation_model():
    return True

def get_sentiment_transformation_model():
    return True

def get_style_transfer_model():
    pass

def get_data_source(test=False):
    if test:
        data = __generate_random_sentences__()
    else:
        data = pd.read_csv('./data/yelp_review_data.csv', names=['Sentences'])
    return data

@st.cache
def init_application():
    sentiment_evaluation_model = get_sentiment_evaluation_model()
    sentiment_transformation_model = get_sentiment_transformation_model()
    style_transfer_model = get_style_transfer_model()
    data = get_data_source()
    return sentiment_evaluation_model, sentiment_transformation_model, style_transfer_model, data
## sentence transformation and evaluation function

#@st.cache
def transform_style(sentences, test = True):
    num_of_sentences = len(sentences)
    if style_transfer_model is None or test:
        sentences = __generate_random_scores__(num_of_sentences)
    else:
        sentences = __transform_styl_with_model__(sentences)
    # else do inference using the evaluation model
    return sentences

#@st.cache
def transform_sentiment(sentences, target_level, test = False):
    num_of_sentences = len(sentences)
    if sentiment_transformation_model is None or test:
        sentences = __generate_random_sentences__(num_of_sentences).iloc[:,0]
    else: # else do inference using the tanformation model
        sentences = __transform_sent_with_model__(sentences, target_level)
    sentences.columns= ['Transformed Sentences']
    return sentences

@st.cache
def sentiment_evaluation_source(sentences, test = False):
    return __sentiment_evaluation__(sentences, test)

@st.cache
def sentiment_evaluation_transform(sentences,  test = True, target=None):
    scores = __sentiment_evaluation__(sentences, test, target)
    scores.columns = ['Transformed Sentiment']
    return scores

## visualization functions

def visualize_word_cloud(data_processed):
    single_sentence = ""
    for sentence in list(data_processed.iloc[:,0]):
        single_sentence += sentence + " "
    wc_img = WordCloud(background_color="white",width=700, height=300).generate(single_sentence).to_image()
    return wc_img

def sentiment_transformation_exploration(source):
    # evaluate origianl data
    sentiment_score = sentiment_evaluation_source(source.loc[:,'Sentences'])
    sentiment_score.column = ['Sentiment']
    mean_score = 0.2*sentiment_score.mean()[0]//0.2
    # select transformation level
    sentiment_transform_level = st.slider('Transform the Sentiment (Higher means more positive)', 1.0, 5.0, value=float(mean_score), step=0.2)
    
    if sentiment_transform_level != mean_score:
        # transform data
        transformed_source = transform_sentiment(source.iloc[:,0], sentiment_transform_level)
        sentiment_score_transformed = sentiment_evaluation_transform(transformed_source, target = sentiment_transform_level)
        # output original text and transformed data
    else:
        # transform data
        transformed_source = source.copy().iloc[:,0]
        transformed_source.columns = ['Transformed Sentences']
        sentiment_score_transformed = sentiment_score.copy()
        sentiment_score_transformed.columns = ['Transformed Sentiment']
        sentiment_score.columns = ['Sentiment']
        
    # output original text and transformed data
    output_data = pd.concat([source, transformed_source], axis = 1)        
    st.write(output_data)        
    graph_data= pd.concat([sentiment_score, sentiment_score_transformed], axis = 1)
    distribution_chart = alt.Chart(graph_data).transform_fold(
        ['Sentiment',
         'Transformed Sentiment'],
        as_ = ['Measurement_type', 'value']
        ).transform_density(
            density='value',
            bandwidth=0.3,
            groupby=['Measurement_type'],
            extent= [1, 5],
        ).mark_area().encode(
            alt.X('value:Q'),
            alt.Y('density:Q'),
            alt.Color('Measurement_type:N')
        ).properties(width=800, height=400)
    st.altair_chart(distribution_chart)

def dataset_exploration(source):
    # evaluate data and preprocess
    sentiment_score = sentiment_evaluation_source(source.loc[:,'Sentences'])
    sentiment_score.columns = ['Sentiment']
    data_processed = pd.concat([source, sentiment_score], axis = 1)
    # speaker = st.selectbox('Select a Speaker from the data', ["All"] + list(source.loc[:,'Speaker'].unique()))
    # if speaker != 'All':
    #     data_processed = data_processed[data_processed.loc[:,'Speaker']==speaker]
    
    # output original text and corresponding sentiment score  
    wc_img = visualize_word_cloud(data_processed)
    st.image(wc_img)
    st.write(data_processed)
    st.subheader("Sentiment Distribution of Data")
    distribution_chart = alt.Chart(data_processed).transform_fold(
        ['Sentiment'],
        as_ = ['Measurement_type', 'value']
        ).transform_density(
            density='value',
            bandwidth=0.3,
            groupby=['Measurement_type'],
            #extent= [1,5],
            extent=[-10,10],
        ).mark_area().encode(
            alt.X('value:Q'),
            alt.Y('density:Q'),
            alt.Color('Measurement_type:N')
        ).properties(width=800, height=300)
    st.altair_chart(distribution_chart)
    pass

def single_sentence_exploration():
    user_input = st.text_area("Input Sentences", "Input Here")
    st.subheader("Your Input")
    st.write(user_input)
    # st.subheader("The input sentence is most similar to:")
    # st.write("Speaker A")
    sentiment_score = int(0.2*sentiment_evaluation_source([user_input]).iloc[0]//0.2)

    sentiment_transform_level = st.slider('Transform the Sentiment (Higher means more positive)', -10, 10, value=sentiment_score, step=1)

    if sentiment_transform_level != sentiment_score and user_input != "Input Here":
        transform_outcome = transform_sentiment([user_input], sentiment_transform_level).iloc[0][0]
        st.subheader('Your Transformed Score')
        transformed_score = int(0.2*sentiment_evaluation_source([transform_outcome]).iloc[0]//0.2)
        st.write(str(transformed_score))
    elif user_input != 'Input Here':
        st.subheader('Your Sentiment Score')
        st.write(str(sentiment_score))
        transform_outcome = user_input
    else:
        transform_outcome = user_input
    st.subheader("Transformation Result")
    st.write(transform_outcome)
    # st.subheader("The transformed sentence is most similar to:")
    # st.write("Speaker B")
    
def visualize_though_model(data):
    # level = st.selectbox('Select in what aspect you want to explore', ['Sentence Level', 'Dataset Level'])
    level = 'Sentence Level'
    if level == 'Dataset Level':
        sentiment_transformation_exploration(data)
    elif level == 'Sentence Level':
        single_sentence_exploration()
    pass

def visualize_dataset(data):
    dataset_exploration(data)
    pass

##main functions

def main_text_style_transfer():
    global sentiment_evaluation_model, sentiment_transformation_model, style_transfer_model, data
    sentiment_evaluation_model, sentiment_transformation_model, style_transfer_model, data = init_application()
    st.title("Exploring the Style of Speech")
    vis_type = st.selectbox('Select the way you want to the speech style', ['Explore With Models','Dataset Overview'])
    if vis_type == 'Dataset Overview':
        visualize_dataset(data)
    elif vis_type == 'Explore With Models':
        visualize_though_model(data)

main_text_style_transfer()










