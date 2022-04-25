import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from PIL import Image
import docx2txt
import os
import textdistance as td
#from PyPDF2 import PdfFileReader
#import pdfplumber

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras import utils 

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk.stem
from nltk.stem import LancasterStemmer, SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
  
tfidf = pickle.load(open('Tf-Idf.sav','rb'))
classifier = pickle.load(open('Final_model.sav','rb'))
  
def welcome():
    return 'welcome all'

def read_resume(file):
    if file is not None:
        file_details = {'File Name' : file.name, 'File Type' : file.type, 'File Size' : file.size}
        st.write(file_details)
        if file.type == 'text/plain':
            raw_text = str(file.read(), "utf-8")
            #st.text(raw_text)
        elif file.type == 'application/pdf':
            try:
                with open(file, 'rb') as pdf:
                    pages = PyPDF2.PdfFileReader(pdf)
                    count = pdfReader.numPages
                    raw_text = []
                    for i in range(count):
                        page = pdfReader.getPage(i)
                        raw_text.append(page.extractText())
                    pdf.close()
            except:
                st.write("None")
        else:
            raw_text = docx2txt.process(file)
            #st.write(raw_text)
        return list(raw_text)


def tfidf_convert(data):
    vec = tfidf.transform([data])
    return vec.toarray()
    
# defining the function which will make the prediction using 
# the data which the user inputs   
def prediction(output):    
    op = ' '.join(output)
    X = tfidf_convert(op)
    prediction = classifier.predict(X)[0]
    return prediction



def create_df():
    directory = 'C:/Users/arund/OneDrive/Desktop/Projects/NLP/Data/Resumes'
    text = ''
    resume_dict = {}
    for a_file in os.listdir(directory):
        content = []
        if a_file.endswith('.docx'):
            text = docx2txt.process(directory + '/' + str(a_file))
            #print(text, '\n\n------------------------------------------------------------------------------------------\n\n')
            resume_dict[str(a_file[:-5])] = text
    resume_data = pd.DataFrame.from_dict(resume_dict, orient  = 'index', columns = ['Resume'])
    resume_data.index.name = 'Names'   

    
    directory = 'C:/Users/arund/OneDrive/Desktop/Projects/NLP/Data/JobDesc'
    text = ''
    job_desc_dict = {}
    for a_file in os.listdir(directory):
        content = []
        if a_file.endswith('.docx'):
            text = docx2txt.process(directory + '/' + str(a_file))
            #print(text, '\n\n------------------------------------------------------------------------------------------\n\n')
            job_desc_dict[str(a_file[:-5])] = text
    job_desc_data = pd.DataFrame.from_dict(job_desc_dict, orient  = 'index', columns = ['Job_Description'])
    job_desc_data.index.name = 'Profile'
    return resume_data, job_desc_data

def clean_data(text): 
    stop_words = stopwords.words('english')
    tokens = word_tokenize(text)
    cleaned_tokens = []

    for tok, tag in pos_tag(tokens):
        tok = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\)]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', tok)
        tok = re.sub("(@[A-Za-z0-9_]+)","", tok)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        tok = lemmatizer.lemmatize(tok, pos)

        if len(tok) > 0 and tok not in punctuation and tok.lower() not in stop_words:
            cleaned_tokens.append(tok.lower())
    return cleaned_tokens

def do_tfidf(token):
    tfidf = TfidfVectorizer(max_df=0.05, min_df=0.002)
    words = tfidf.fit_transform(token)
    sentence = " ".join(tfidf.get_feature_names_out())
    return sentence

def match(resume, job_des):
    j = td.jaccard.similarity(resume, job_des)
    s = td.sorensen_dice.similarity(resume, job_des)
    c = td.cosine.similarity(resume, job_des)
    o = td.overlap.normalized_similarity(resume, job_des)
    total = (j+s+c+o)/4
    # total = (s+o)/2
    return total*100

# this is the main function in which we define our webpage 
def main():
    menu = ['Get Profiles based on your requirements', 'Predict your Role/Position with Resume']
    st.sidebar.title("Hello! Welcome to our website")
    choice = st.sidebar.radio('Select from the options', menu)
    if choice == 'Predict your Role/Position with Resume':
        # giving the webpage a title
        #st.title("Job Position Prediction")
          
        # here we define some of the front end elements of the web page like 
        # the font and background color, the padding and the text to be displayed
        html_temp = '''
        <div style ="background-color:lightskyblue;padding:13px">
        <h1 style ="color:black;text-align:center;">Job Position Prediction</h1>
        </div>
        '''
          
        # this line allows us to display the front end aspects we have 
        # defined in the above code
        st.markdown(html_temp, unsafe_allow_html = True)
          
        # the following line create file upload widget in which the user can upload file
        # the data required to make the prediction
        label = 'Add your resume below!'
        file = st.file_uploader(label, type=['docx'], accept_multiple_files=False) 
        result =""
          
        # the below line ensures that when the button called 'Predict' is clicked, 
        # the prediction function defined above is called to make the prediction 
        # and store it in the variable result
        if st.button("Predict"):
            result = prediction(read_resume(file))
        st.success('The output is {}'.format(result))
    
    else : 
        # here we define some of the front end elements of the web page like 
        # the font and background color, the padding and the text to be displayed
        html_temp = '''
        <div style ="background-color:lightskyblue;padding:13px">
        <h1 style ="color:black;text-align:center;">Get Candidates suitable for the position </h1>
        </div>
        '''
          
        # this line allows us to display the front end aspects we have 
        # defined in the above code
        st.markdown(html_temp, unsafe_allow_html = True)
        
        # Create dataframe
        resume_data, job_desc_data = create_df()
        
        resume_data['Clean_data'] = resume_data['Resume'].apply(lambda x:clean_data(x))
        job_desc_data['Clean_data'] = job_desc_data['Job_Description'].apply(lambda x:clean_data(x))
        
        
        # Fit tfidf
        resume_data['TF_IDF_Based'] = resume_data['Clean_data'].apply(lambda x: do_tfidf(x))
        job_desc_data['TF_IDF_Based'] = job_desc_data['Clean_data'].apply(lambda x: do_tfidf(x))
        
        for i in job_desc_data.index:
            resume_data['Scores for ' + str(i)] = resume_data['TF_IDF_Based'].apply(lambda x:match(x, job_desc_data.TF_IDF_Based[i])) 
        
        
        if len(job_desc_data.index) <= 1:
            st.write(
                "There is only 1 Job Description present. It will be used to create scores.")
        else:
            st.write("There are ", len(job_desc_data.index),
                     "Job Descriptions available. Please select one.")
        
        role = st.selectbox("Select the role", [i for i in job_desc_data.index])
        option_yn = st.selectbox("Show the Job Description ?", options=['YES', 'NO'])
        if option_yn == 'YES' and (st.button('Show Job Description')):
            st.markdown("---")
            st.markdown("### Job Description :")
            st.info(job_desc_data['Job_Description'][role])
        
        num = st.slider("Select number of profiles you need", 1, len(resume_data.index))
        if(st.button('Submit')):
            sorted_resume = resume_data.sort_values(by=['Scores for ' + str(role)], ascending=False)
            st.dataframe(sorted_resume[['Scores for ' + str(role)]][:num])
            st.balloons()
            
            
if __name__=='__main__':
    main()