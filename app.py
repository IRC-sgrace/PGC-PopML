# import packages
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# import streamlit
import streamlit as st
import joblib as jl
from sklearn.ensemble import RandomForestClassifier

# Web app title
st.title('Welcome to Pop Gen!')
st.subheader("Your user-friendly Machine Learning App!")
# describe the web app
st.write("This app helps wildlife forensic scientists locate the origin of your sample. Contact our team for all your population assignment needs!")
st.text("Pop Gen is currently trained on Sea Cucumber dataset - diploid organisms i.e. two alleles")
st.text("For more info on Population Genetics, contact Sir Carlo Lapid")


# read the data
#data = pd.read_csv('hscabra_knnimpute_15pops.csv')

#display data
#if st.checkbox('Show raw data'):
#    st.subheader('Raw data')
#    st.write(data)

# draw a histogram
st.subheader('Take a look at this cute, squiggly Sea cucumber!')

from PIL import Image
image = Image.open('hscabra_pic.jpg')
st.image(image, caption='#protectatallcosts #nationaltreasure',
          use_column_width=True)

st.subheader('Input your sample\'s genetic data e.g. fragment length for each loci.')
st.text('If the allele is null for that specific loci, set the slider to zero.')

st.title('Population Assignment Generator')

feat1 = st.sidebar.slider('Hsc 40',0,356,260)
feat2 = st.sidebar.slider('Hsc 40.1',0,376,276)
feat3 = st.sidebar.slider('Hsc 24',0,219,205)
feat4 = st.sidebar.slider('Hsc 24.1',0,223,207)
feat5 = st.sidebar.slider('Hsc 20',0,240,214)
feat6 = st.sidebar.slider('Hsc 20.1',0,244,218)
feat7 = st.sidebar.slider('Hsc 11',0,164,150)
feat8 = st.sidebar.slider('Hsc 11.1',0,170,154)
feat9 = st.sidebar.slider('Hsc 59',0,260,236)
feat10 = st.sidebar.slider('Hsc 59.1',0,264,236)
feat11 = st.sidebar.slider('Hsc 28',0,423,240)
feat12 = st.sidebar.slider('Hsc 28.1',0,492,276)
feat13 = st.sidebar.slider('Hsc 44',0,206,198)
feat14 = st.sidebar.slider('Hsc 44.1',0,238,198)
feat15 = st.sidebar.slider('Hsc 17',0,222,192)
feat16 = st.sidebar.slider('Hsc 17.1',0,238,192)
feat17 = st.sidebar.slider('Hsc 48',0,162,154)
feat18 = st.sidebar.slider('Hsc 48.1',0,190,154)
feat19 = st.sidebar.slider('Hsc 42',0,255,251)
feat20 = st.sidebar.slider('Hsc 42.1',0,285,253)
feat21 = st.sidebar.slider('Hsc 49',0,291,159)
feat22 = st.sidebar.slider('Hsc 49.1',0,291,167)
feat23 = st.sidebar.slider('Hsc 31',0,175,175)
feat24 = st.sidebar.slider('Hsc 31.1',0,175,175)
feat25 = st.sidebar.slider('Hsc 62',0,268,250)
feat26 = st.sidebar.slider('Hsc 62.1',0,274,250)

if feat1 or feat2 == 0:
    isnull40 = 1
else: isnull40 = 0
    
if feat3 or feat4 == 0:
    isnull24 = 1
else: isnull24 = 0

if feat5 or feat6 == 0:
    isnull20 = 1
else: isnull20 = 0
    
if feat7 or feat8 == 0:
    isnull11 = 1
else: isnull11 = 0
    
if feat9 or feat10 == 0:
    isnull59 = 1
else: isnull59 = 0
    
if feat11 or feat12 == 0:
    isnull28 = 1
else: isnull28 = 0
    
if feat13 or feat14 == 0:
    isnull44 = 1
else: isnull44 = 0
    
if feat15 or feat16 == 0:
    isnull17 = 1
else: isnull17 = 0
    
if feat17 or feat18 == 0:
    isnull48 = 1
else: isnull48 = 0
    
if feat19 or feat20 == 0:
    isnull42 = 1
else: isnull42 = 0
    
if feat21 or feat22 == 0:
    isnull49 = 1
else: isnull49 = 0
    
if feat23 or feat24 == 0:
    isnull31 = 1
else: isnull31 = 0

if feat25 or feat26 == 0:
    isnull62 = 1
else: isnull62 = 0


# if 'Hsc 40' or 

st.text("This section will output the Predicted Population in real-time.")

st.text("The available models at the moment include fastAI, Gradient Boosting Machine, TPOT, etc.")
st.text("In the meantime, let's run the Random Forest Classifier, live!")


saved_model = jl.load('popgen_model.sav')

import time
'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'

predicted_population = saved_model.predict([[feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8,feat9,feat10,feat11,feat12,feat13,feat14,feat15,feat16,feat17,feat18,feat19,feat20,feat21,feat22,feat23,feat24,feat25,feat26,isnull40,isnull24,isnull20,isnull11,isnull59,isnull28,isnull44,isnull17,isnull48,isnull42,isnull49,isnull31,isnull62]])[0]


if predicted_population==1:
    out = 'Sta. Ana, Cagayan (PH Sea)'
elif predicted_population==2:
    out = 'Sorsogon, Sorsogon (PH Sea)'
elif predicted_population==3:
    out = 'Guiuan, Eastern Samar (PH Sea)'
elif predicted_population==4:
    out = 'Masinloc, Zambales (West PH Sea)'
elif predicted_population==5:
    out = 'El Nido, Palawan (Sulu Sea)'
elif predicted_population==6:
    out = 'Romblon (Internal Seas)'
elif predicted_population==7:
    out = 'Concepcion, Iloilo (Internal Seas)'
elif predicted_population==8:
    out = 'Tigbauan, Iloilo (Internal Seas)'
elif predicted_population==9:
    out = 'Cebu (Internal Seas)'
elif predicted_population==10:
    out = 'Bohol (Internal Seas)'
elif predicted_population==11:
    out = 'Dumaguete (Internal Seas)'
elif predicted_population==12:
    out = 'Coron, Palawan (Sulu Sea)'
elif predicted_population==13:
    out = 'Sta. Cruz, Davao (Celebes Sea)'
elif predicted_population==14:
    out = 'Maasim, Sarangani (Celebes Sea)'
else: 
    out = 'Bongao, Tawi-Tawi (Celebes Sea)' # for pop = 15


st.subheader('The location of origin of the input sample is: ')
st.title(out)


@st.cache
def get_data():
    return pd.read_csv("ph.csv")
df = get_data()
st.header("Where were the Holothuria scabra samples taken?")
st.subheader("On a map")
st.markdown("The following map shows where the sea cucumber samples used in population assignment were taken.")
st.map(df,zoom=6)
st.text('Map coordinates imported from https://simplemaps.com/data/world-cities')

st.code("""
@st.cache
def get_data():
    return pd.read_csv("ph.csv")
df = get_data()st.map(df,zoom=6)
""", language="python")

with st.spinner('Wait for it...'):
    time.sleep(5)
    
st.success('Done!')

st.markdown("**THE TEAM**")
st.markdown("DJ, Theoretical Statistician | NJ, Forensic Geneticist | Shane, Data Scientist & Software Developer ")

st.markdown('In collaboration with')
from PIL import Image
image_2 = Image.open('FTWLOGO.jpg')
st.image(image_2, caption='Free Data Science scholarship program for women! Help us turn PH into the data capital of the world!',
          use_column_width=True)

st.markdown("## Party time!")
st.write("Yay! You're done with population assignment. Click below to celebrate.")
btn = st.button("Celebrate!")
if btn:
    st.balloons()
