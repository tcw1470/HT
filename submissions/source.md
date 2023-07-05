# %% [markdown]
# # *Proglog*
# 
# *Over the past few years, global attention and resources have been allocated to urgent needs from (international) wars and vaccination development/dissemination efforts, leaving many humanitarian crises untackled. As we transition into post-pandemic times, I hope my article below will solicit your attention to help combat human trafficking with your machine learning and data analytics expertise. Your input of any kind is greatly appreciated and I sincerely thank you all in advance for investing your time and interests to skimming through my draft.*
#     
# *Keywords: child-, adult-, women-, and human trafficking; machine learning solutions*
#     
# <div style="background-color:black; padding:1em" >
#     <img src='https://www.bluebutterflyfoundation.org/images/TraffickedGirlFace.jpeg' width="380px"/>
#     <img src='https://www.bluebutterflyfoundation.org/images/TraffickedBoys.jpeg' width="400px"/>
#     <BR>
#     <p style="color:gray">Image source: <a href='https://www.bluebutterflyfoundation.org/child_trafficking_nepal.php'>https://www.bluebutterflyfoundation.org/child_trafficking_nepal.php</a>
#     </p>
# </div>
# 
# # Acknowledgements
# 
# This notebook gratefully acknowledges authors of the following notebooks:
# - [Research Paper Recommendation](https://www.kaggle.com/code/harshsingh2209/research-paper-recommendation)
# - [EDA on the Global Human Trafficking dataset](https://www.kaggle.com/code/viktorpolevoi/global-human-trafficking-eda)
#  - The code in this notebook was mostly mirrored, with coded adjusted near the end so that the two figures mentioned in my article expresses data in percentages instead of absolute numbers
#     
# <hr>
#     
#     
# # Introduction <a class="anchor" id="intro"></a>
# 
# Human trafficking, a serious violation of our basic human rights, is a global concern that impacts all social classes, demographics, gender, and races. Estimates from the International Labour Organization made in 2016 suggest that more than five people across the globe fall prey to human trafficking each day [(Toney-Butler et al., 2023)](#ref_defn), affecting more than forty three million families worldwide, with 72% of the victims being women and girls. Newer statistics estimate that *forced labour* had victimized 25 million people and *forced marriages* victimized another 15.4 million individuals globally. Unfortunately, [most of these statistics are underestimated](#ref_stats) as it remains unknown the survival rates of victims trafficked for [forced begging](#ref_beg), child soldiers, organ removals, etc.
#     
# The physical and psychological harms done to victims are perminent, if not fatal. Survivors of forced labour were studied by [(Gezie et al. 2021)](#ref_harm) to analyze the extents of their physical health after their rescues. Among the 1,387 study participants who transited through the corridors of Ethiopia for forced labour, 37% of them got exploited in housework; 27% in animal and agriculture farming; 18% in services and manufacturing industries. After they were rescued, a majority of them reported health problems as urinary tract infections, gynaecological problems, bone fracture, back pain, stomachache, forgetfulness, etc. 
# 
# From an economic standpoint, human trafficking is a $150 billion business that plagues all types of sectors, including forced labour in beauty industries, entertainment, hospitality, construction, agriculture, commercial fishing. The regions are vast; forced labour in commercial fishing alone happening in the United States, United Arab Emirate, Thailand, Taiwan, Sudan, Philippines, Ireland, etc. These criminal recruitments are done systematically via the following elements that collectively described in what is known as the **A-M-P model** [(Dimas et al. 2022)](#ref_analytics):
# - **A**ctions that perpetrators take to induce (recruits, harbors, transports, provides, etc.)
# - **M**eans of "force, fraud, or coercion" [(Toney-Butler et al.)](#ref_defn) they use
# - **P**urposes of illegal recruitment for commercial sex or labour trafficking.
# 
# At the same time, vulnerability risk factors include adverse childhood experiences and/or any prior experiences of sexual abuse, sexual assault, domestic violence, gang violence, community-violent exposures, as well as the lack of education, awareness, cultural exploitations. Traffickers do not discriminate by age, gender, race, nor social class and recruit *anywhere*. While predominantly in developing countries and countries at war, the European Council's website presents countless stories of survivors from developed countries, cases include teens from Netherlands who got seduced by their partners into prostitution within their home countries, and transiting foreigners who got "detained" and forced into domestic service when their passports got confiscated in countries such as France.
# 
# The search for solutions to rescue victims is immensely difficulty but thankfully are in active pursuits. In  US, legislation combats human trafficking with the **3P's: protect, prosecute, and prevent**. Scholars [(Dimas et al. 2022)](#ref_analytics) add **partnership** to emphasize the critical importance of collaborations across multiple disciplines as healthcare providers, social workers, prosecutors, police, etc. The European Union's four-year strategy also allocated 13 million European dollars since 2021 to disrupt the online and offline business model of traffickers. 
# 
# Data from various meta reviews [(Athreya, 2020)](#ref_econ), [(Milivojevic et al. 2021)](#ref_tech) show that technologies to combat trafficking include:
# - **Resue**: time-sensitive and long-term rescues
# - **Tracking**: Identify sources of slavery perpetrators and track their activities
# - **Prosecution**: Punish criminals using technologies that lead to successful prosecution 
# 
# In this article, we will examine ***how Kaggle and the machine learning communities are combatting this humanitarian criis along some of these fronts***. We will also investigate how technological advances are being exploited by traffickers and suggest future directions researchers and experts like many of you may explort to assist in this constant race for victory. 
# 
# <style>
# div {
#     display: list-item; 
# }
# mark{
#     color: red;
# }</style>
# 
# <a type='anchor' id='toc'><a/>
# Contents:
# - [Introduction](#intro)
# - [Methods](#methods)
# - [Results](#res)
# - [Discussions](#discuss)
# - [References](#refs)
# - [Supplementary data](#suppl)
# 
# 
# 
# # Methods <a type='anchor' id='methods'><a/>
#  
# ## Collection of study materials
# 
# Using ```urllib``` and ```BeautifulSoup```, we searched for articles on *Research Square* and the *Journal of Anti Trafficking Review* using the following keywords:
# - human trafficking
# - women trafficking
# - child trafficking
# - kaggle
# - artificial intelligence, machine learning, data mining
# 
# We conducted similar searches on Kaggle's grey literature:
# - Discussion
# - Datasets
# - Competitions 
# 
# 
# # Results <a type='anchor' id='results'><a/>
# 
# ## Overview on operational research and data analytics
# 
# Dimas et al. [(2022)](#ref_analytics) compiled a list of 140+ articles that employed operational research and data analytics to fight human trafficking. Their inclusion criteria are: 
# - Application or case study was anti-trafficking efforts
# -  Studies, theses, dissertations
# - Themes one or more on: position/ thoughts, analytical methods, and operations research methods 
# - Articles published during 2010 - 2021, inclusive, at the time of their article's publication
#  
# They then grouped the key themes discussed in these articles into 21 categories, some of which include:
# 
# - Active learning |  Supervised learning, unsupervised learning/clustering 
# - Natural language processing, web scraping
# - Data envelopment analysis, facility location, simulation, game theory 
# - Link inference, social network analysis, investigative search
# 
# 
# In the Supplementary, we provide some contexts and examples of their anti-trafficking applications.

# %% [markdown]
# 
# ## Contributions from the Kaggle and ML communities
# 
# Contributions from Kaggle include:
# 1. Advances in the Rescue efforts
# 2. Dissemination of datasets
# 2. Indirect remedies via analytic competitions 
# 
# 
# ### 1. Direct: towards rescue efforts
# 
# #### Visual Kinship Recognition challenge 
# 
# Kaggle's *Recognizing Faces in the wild* competition held in 2019 asked competitors to develop a model that determines if two faces from two photos are biologically related.
# 
# #### Rescues of victims via facial and hotel room recognitions 
# 
# On the **tracking** front, the most relevant competitions held are the "Hotel-ID to combat human trafficking" competition series held in 2021 and 2022. 
# 
# Often, victims of sex trafficking were photographed in hotel rooms for promotional purposes [Stylianou et al. 2019](#ref_hotel50k). These photos were taken at uncommon camera angles in poor lighting scenarios, leading to images of especially low quality. Therefore, these Kaggle competitions asked competitors to develop methods that would accurately identify the hotels pictured in these hotel images. The challenge provided a training set $\mathcal{T}_\text{challenge}$. To evaluate the submitted solutions, the mean average precision @ 5 (MAP@5) is used to evaluate the submitted solutions: that is, the five most probable hotel names predicted by the method will be considered in a weighted manner.  
# 
# 
# In 2021, the winning solution by [Mr. Ozaki](https://ho.lc/), a Kaggle Master (software engineer and scholar), approached this problem with the following strategies:
# 1. **Data curation**: trained a classification model using the Hotels50k dataset  ($\mathcal{T}_\text{new}$) that would identify the angle needed to correct the rotated views in 4,859 of the 97,554 photos in the training set
# 2. **Dataset expansion**: expand $\mathcal{T}_\text{challenge}$ by adding a subset of images from $\mathcal{T}_\text{new}$ that were similar to those in $\mathcal{T}_\text{challenge}$ 
# 	- Similarity was determined by the nearest neighbour 
# 3. **Label-constrained data augmentation**
# 4. Metric learning and nearest neighbour search
# 
# Based on results from his ablation study, Mr. Ozaki concluded that the three architectures that he explored (namely, ResNeSt101e, RegNetY120, and Swin Transformer) yielded minimal differences in terms of performance improvements as compared to the use of a carefully curated and expanded training set of hotel images.
# 
# In 2022, Kaggle Master [Mr. Austin](https://www.linkedin.com/in/david-austin-037630123) examined the hotel room images and noted the following observations: 
# - Images were occluded by the subjects (victims)
# - Many vertical and horizontal features repeated against backgrounds
# 
# To address the problem with occlusions, Mr. Austin impainted the occluded image regions via several approaches (use of generative adversarial networks, traditional computer vision impainting methods) and proposed the *BlendFlip algorithm* whose objective is to maintain the visual cohesion of the recovered images.
# 
# Inspired by Mr. Austin's ideas and those presented by the forth top-winner, US scholars [Lin et al. 2022](#ref_lin) formulate hotel recognition as a **metric learning task** whereby the similarity between hotel images are learned, rather than pre-defined as done in the aforementioned methods. Using a *triplet loss*, their framework learns image features that will cluster in the feature space if they were extracted from photos taken from the same hotel and that images from different hotels will form different clusters with great inter-cluster distances. They also conducted ablation studies to benchmark different backbones and conclude that EfficientNet-B5 was superior to ConvNeXt, ResNet50 in terms of MAP@5. 
# Lin et al. also employ random erasing, random flip, random crop for data augmentation. Using saliency maps to internally validate their results, they found: 
# - Furnitures such as beds and televisions are rarely important; textures such as walls, curtains, vanities are; 
# - Lamps have shapes unique to hotels (and their levels of budget); tables and chairs occasionally have similar effects  
# 
# ### 2. Dissemination of related datasets & exploratory analyses published in users' notebooks
#  
# In Kaggle, there are over 45 datasets on crime; the most relevant (interesting) ones are listed in the table below. 
#   
# | Name | Source | High level info  | Misc. notes |
# | :-- | :-- | :-- | :-- |
# | Global Human Trafficking (GHT) | Counter-Trafficking Data Collaborative | Data on 48.8k cases | Missingness is indicated as -99 |
# | Human Freedom Index | [Vasquez et al. 2022](#ref_freedom) | Annual 2018-2022 | women-specific attributes |
# | [UK human trafficking (UKHT) ](https://www.kaggle.com/datasets/algosforgood/uk-human-trafficking-data) |  UK  National Crime Agency (NCA) | Reports from NCA website 2013-2016  | [Notebook visualizing the reported incidents over time](https://www.kaggle.com/code/paultimothymooney/sample-submission-for-january-s-featured-task) | 
# | [FBI human trafficking Database](https://www.kaggle.com/datasets/larsen0966/human-trafficking-fbi-data)| | Data from crime reports from 2014 - 2017 | Data records offenses and arrests that may not be "interpreted as definitive statement of human trafficking" <BR> [An exploratory analysis notebook](https://www.kaggle.com/code/larsen0966/is-human-trafficking-really-a-problem) |
# | San Francisco  Crime Classification | San Francisco OpenData | Data from crime reports from 2003-2015 | - Classification of criminal activities such as prostitution, sex offenses, pornography, kidnapping, missing person, etc. <br>- [Tableau dashboard promoted by a Kaggle user](https://public.tableau.com/app/profile/jaysha101/viz/CRIMEINSANFRANCISCO/SFCrimeDashboard) |
# | [Denver Crime Data](https://www.kaggle.com/datasets/paultimothymooney/denver-crime-data) | National Incident Based Reporting System  | Data from crime reports since June 2017 | - Dynamically updated  ~~annually~~ monthly such that corrections may be made in subsequent version(s) <BR> - [Shinny app](https://dhallsk8.shinyapps.io/denver-crime-data_appTest7/) promoted by a Kaggle user | 
# | Georgia crime by country 2019 - 2021 | [Georgia's Uniform crime reporting program](https://gbi.georgia.gov/services/crime-statistics)| | |
# | [Crime in India](https://www.kaggle.com/datasets/rajanand/crime-in-india)| Updated by community user in 2017 | "state-wise data from 2021" [ibid] | [Authenticity of this dataset has been questioned by a Kaggle user](https://www.kaggle.com/datasets/rajanand/crime-in-india/discussion/383361?sort=votes&select=20_Victims_of_rape.csv) | 
# | [Early Russian news on drug addiction](https://www.kaggle.com/datasets/dadalyndell/early-russian-news-articles-on-drug-addiction)| Kaggle user | News and other publications from 1980-1996 that mentioned illegal drugs and drug-users| - Posted in March 2023 <BR> - Not been explored by other Kaggle users |  
# 
# 
# 

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-07-05T03:54:43.470199Z","iopub.execute_input":"2023-07-05T03:54:43.470692Z","iopub.status.idle":"2023-07-05T03:54:43.611318Z","shell.execute_reply.started":"2023-07-05T03:54:43.470657Z","shell.execute_reply":"2023-07-05T03:54:43.609507Z"}}
import os, sys
from IPython.core.display import display, HTML
from pathlib import Path
import matplotlib.pyplot as plt
import IPython
from IPython.display import display
import tensorflow as tf; import tensorflow_hub as hub
from tensorflow.keras.losses import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
#!rm -fR /kaggle/working/*
#!pip install -U kaleido #--target==kaggle/working/site-packages/ # to export figures from plotly

import pandas as pd
import numpy as np

import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
import warnings; warnings.filterwarnings('ignore')

# !pip install --quiet pycountry_convert
# from pycountry_convert import country_alpha2_to_country_name, country_name_to_country_alpha3

import umap # order the areas/ themes
# https://www.kaggle.com/discussions/product-feedback/43354
from plotly.offline import init_notebook_mode; init_notebook_mode(connected=True);

sheet = '2022_Operations Research and An'
#xlsm_df=pd.read_excel('https://github.com/gldimas/Dimas-et.-al-2022_Human-Trafficking-Literature-Review/raw/main/Updated%20Versions%20of%20Data%20and%20Dashboard/Current%20Versions%20of%20Data%20and%20Dashboard/08_29_2022_ImprovedMacro_DetailedAnalysis_2022_Operations%20Research%20and%20Analytics%20to%20Combat%20Human%20Trafficking%20A%20Review%20of%20Academic%20Literature_Data.xlsm', sheet_name=sheet)

xlsm_df = xlsm_df.rename('Prosection', 'Prosecution' )
solution_fronts = ['Prevention', 'Protection', 'Prosecution','Partnership'] 
regions  = ['North America', 'South America','Europe', 'Asia', 'Africa', 'Australia/Oceania', 'Unspecified/All regions']

subthemes = ['Decision Support', 'Inferential Statistics / Detection',' Network Flow', 'Resource Allocation', 'Supply Chain', 'Other / Unspecified']
flds = ['Network Flow', 'Resource Alloction',
           'Supply Chain', 'Decision Support', 'Other / Unspecified',
           'Inferential Statistics / Detection', 
           'Data Envelopment Analysis', 'Network Interdiction',
           'Integer Programming', 'Facility location', 'Queuing Theory',
           'Simulation', '(Social) Network Analysis', 'Empirical Analysis',
           'Active Learning', 'Graph Construction', 'Link Inference',
           'Machine / Deep Learning (General)', 'Clustering or Classification',
           'Unsupervised or Minimally Supervised Learning',
           'Natural Language Processing', 'Information Extraction',
           'Investigative Search', 'Game Theory', 'Other',
           'Network / Graph Theory', 'Web Crawling / Scraping'] 

df=xlsm_df[flds] 
#my_umap = umap.UMAP(n_components=1); fitted = my_umap.fit(df.to_numpy()); 
#xlsm_df[ ['umap_order']]= my_umap.transform(df.to_numpy()); 

# =========== figure 1 =========== 
fig = go.Figure();
l1='Year of publication'
l2='ML/ data mining areas'
clrs=['red','green','blue','purple', 'yellow', 'black', 'lightgreen']


from plotly.validators.scatter.marker import SymbolValidator
raw_symbols = SymbolValidator().values

raw_symbols= ['','star', 'square', 'triangle-down-open', 'bowtie-open','x', 'triangle-up-open', 'hourglass', 'diamond-x']
for i,r in enumerate(regions):
    q = np.where( np.asarray( xlsm_df[[r]] ) )[0] 
     
    y=xlsm_df.loc[q,'Category'].values    
    #y=xlsm_df.loc[q,'umap_order'].values    
    x=xlsm_df.loc[q,'Year Published'].values *1.        
    x+= np.random.random( len(q) )*0.5               
    z=xlsm_df.loc[q,flds]         
    
    dfn = pd.DataFrame( {l1:x, l2:y, 'area': r })
    dfn= pd.concat( [dfn, z ] )         
    if i==0:
        fig=px.scatter( dfn, x=l1, y=l2, )        
        fig.update_traces( marker=dict( size=15, symbol='cross' ),opacity=.7 )
    else:               
        p=go.Scatter( x=x, y=y, mode='markers', marker_symbol= raw_symbols[i], name=r, fillcolor=clrs[i] ) 
        fig.add_trace(p);
    fig.update_traces( marker=dict( size=15,  ),opacity=.7 )
    
fig.show();
#plt.yticks( np.arange(len(solution_fronts)), solution_fronts)
plt.savefig('/kaggle/working/Dimas2022_lit_location.png' );





# =========== figure 2 =========== 
df=xlsm_df.loc[:,['PaperName','Year Published','Category']+ solution_fronts + flds ]    
df= df.sort_values( by=['Year Published','PaperName' ])
df[['size']] = 0 
for i,r in df.iterrows():
    q=np.where( r[solution_fronts] )[0]
    df['solution fronts']= '; '.join( np.array(solution_fronts)[q] ) 
    q=np.where( r[flds] )[0]
    df['fields']= '; '.join( np.array(flds)[q] )     
    df.loc[i, 'size'] = len( q)
df=df.reset_index()
df['yorder'] = df.index + np.random.random( df.shape[0] )*1
df[['size']] = df[['size']]**2
fig=px.scatter( df, y='Year Published', x='yorder',size='size', color='Category',hover_data=['PaperName','fields', 'solution fronts'] )
#fig.update_traces( hovertemplate='z2:%{customdata[1]:.3f} <br>z3: %{customdata[2]:.3f} '  )
fig.show();
plt.savefig('/kaggle/working/Dimas2022_lit.png' );





# =========== figure 3 =========== 
solutions_area = np.zeros( ( 4, len(flds) ))
for i in range(xlsm_df.shape[0]):    
    Q1= np.where( xlsm_df.loc[i, flds ] )[0]
    Q2= np.where( xlsm_df.loc[i, solution_fronts ] )[0]   
    for q1 in Q1:  
        for q2 in Q2:
            solutions_area[ q2, q1 ]+=1    
df=pd.DataFrame( solutions_area, columns = flds, index=solution_fronts )
fig=px.imshow(df, text_auto=True, aspect='auto', labels={'x':'ML/ Data mining areas of research/ application','y':'Solution fronts'}); 
fig.show();           
plt.savefig('/kaggle/working/Dimas2022_heatmap.png' );

# %% [markdown]
# # Discussions

# %% [markdown]
# 
# #### Closer look at GHT
# 
# Of note is the GHT dataset deposited by the *Counter-Trafficking Data Collaborative* (CTDC), the world's first data portal on this issue. CTDC suggests that the dataset could be used to conduct exploratory analysis to help understand the demographics of the victims. The time series data in GHT dataset may also be used to build models that can predict and/or identify trafficking activities. Preliminary exploratory analyses by [Ms. Satvik](https://www.kaggle.com/code/satvik7/time-series-forecasting-using-arima-model), a student from India, reveal that a majority of victims archived in this database came from Philippines, Ukraine, and Republic of Moldova, and that data entries peaked at 2016 due to the integration of data from the Philippines that year (figures below were borrowed from Satvik's notebook). 
# <div style="display: table;">
# <img style="float: left; width: 40%; align: top" src='https://github.com/tcw1470/HT/assets/138358213/8e477c37-0322-433c-b22d-b1b3365b316e' />
# <img style="float: left; width: 40%; align: top" src='https://github.com/tcw1470/HT/assets/138358213/4aef129a-1821-4320-9950-dd4cc5e7f2cb' />
# </div>
# 
# Another user [Mr. Polevoi](https://www.kaggle.com/code/viktorpolevoi/global-human-trafficking-eda) employed interactive graphs to examine age and gender of the victims. The means of control experienced by a majority of female victims were dominated by psychological abuse, physical abuse, threats, psychoactive substances, and restricted movements. Male victims also reported similar means of control (Figure below), but with lower incidence rates than those of female. There were no gender differences for other types of abuses such as withholding (important) documents, false promises, etc. A majority of the victims (>15.3k) recorded in this database was of female gender who reported of sexual exploitation. The sexual exploits were predominantly distributed to prostitution (95.6%), with 3.4% to pornography and 1% to private sexual service locations. There was no significant differences in gender in forced labour. Another depressing observation is that the predators of the female victims were intimate partners, family members, and friends (Figure *Victim-Predator Relationship*). 
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-07-05T03:50:37.581130Z","iopub.status.idle":"2023-07-05T03:50:37.583172Z","shell.execute_reply.started":"2023-07-05T03:50:37.582925Z","shell.execute_reply":"2023-07-05T03:50:37.582950Z"}}
'''
df = pd.DataFrame( { 'Article counts': [3,18,2,16,54,5,51], 'regions':['Africa','Asia', 'Australia/Oceania' ,'Europe', 'North America', 'South America', 'All other regions']  })
fig=px.pie( df, values='Article counts', names='regions' )
fig.show()

'''
data = pd.read_csv('../input/global-human-trafficking/human_trafficking.csv')
data.replace('-99', np.nan, inplace=True)
data.replace(-99, np.nan, inplace=True)

def get_alpha3(col):
    try:
        iso_3 =  country_name_to_country_alpha3(col)
    except:
        iso_3 = 'Unknown'
    return iso_3

def get_name(col):
    try:
        name =  country_alpha2_to_country_name(col)
    except:
        name = 'Unknown'
    return name

data['meansOfControlConcatenated'] = data['meansOfControlConcatenated'].str.replace('Abuse', 'abuse', regex=True)
data_bar_f = data[(data.meansOfControlConcatenated.notna()) & (data.gender == 'Female')].meansOfControlConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0)
data_bar_m = data[(data.meansOfControlConcatenated.notna()) & (data.gender == 'Male')].meansOfControlConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0)

fig = go.Figure(data=[
    go.Bar(name='Female', x=data_bar_f.index, y=data_bar_f),
    go.Bar(name='Male', x=data_bar_m.index, y=data_bar_m)
])
fig.update_traces(texttemplate='%{value}', textposition='outside')
fig.update_layout(hovermode='x', title_text='Means of Control')
fig.show()



if 0:
    table2 = pd.DataFrame()
    for i in data[data.ageBroad.notna()].ageBroad.unique():
        age_col = pd.DataFrame(data[(data.meansOfControlConcatenated.notna()) & (data.ageBroad == i)].meansOfControlConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0))
        age_col.rename(columns={0: i}, inplace=True)
        table2 = pd.concat([table2,age_col],axis=1)

    age_list = ['0--8', '9--17', '18--20', '21--23', '24--26', '27--29', '30--38', '39--47', '48+']
    table2 = table2.reindex(columns=age_list)

    table2.fillna(0).style.background_gradient(cmap=cm).format('{:,.0f}')

    data_bar_f = data[(data.typeOfExploitConcatenated.notna()) & (data.gender == 'Female')].typeOfExploitConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0)
    data_bar_m = data[(data.typeOfExploitConcatenated.notna()) & (data.gender == 'Male')].typeOfExploitConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0)

    fig = go.Figure(data=[
        go.Bar(name='Female', x=data_bar_f.index, y=data_bar_f),
        go.Bar(name='Male', x=data_bar_m.index, y=data_bar_m)
    ])
    fig.update_traces(texttemplate='%{value}', textposition='outside')
    fig.update_layout(title_text='Type of Exploit')
    fig.show()


    table3 = pd.DataFrame()
    for i in data[data.ageBroad.notna()].ageBroad.unique():
        age_col = pd.DataFrame(data[(data.typeOfExploitConcatenated.notna()) & (data.ageBroad == i)].typeOfExploitConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0))
        age_col.rename(columns={0: i}, inplace=True)
        table3 = pd.concat([table3,age_col],axis=1)

    table3 = table3.reindex(columns=age_list)
    table3.fillna(0).style.background_gradient(cmap=cm).format('{:,.0f}')

    data_sex_type = data.typeOfSexConcatenated.value_counts()

    fig = px.pie(data_sex_type, values=data_sex_type, names=data_sex_type.index,
                title="Distribution of Sex Exploit")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.show()

data_bar_f = data[(data.RecruiterRelationship.notna()) & (data.gender == 'Female')].RecruiterRelationship.apply(lambda x: pd.value_counts(str(x).split("; "))).sum(axis = 0)
data_bar_m = data[(data.RecruiterRelationship.notna()) & (data.gender == 'Male')].RecruiterRelationship.apply(lambda x: pd.value_counts(str(x).split("; "))).sum(axis = 0)

fig = go.Figure(data=[
    go.Bar(name='Female', x=data_bar_f.index, y=data_bar_f),
    go.Bar(name='Male', x=data_bar_m.index, y=data_bar_m)
])
fig.update_traces(texttemplate='%{value}', textposition='outside')
fig.update_layout(title_text='Victim-Predator Relationship')
fig.show()

df=pd.DataFrame( { 'f':data_bar_f/data_bar_f.sum()*100, 'm':data_bar_m/data_bar_m.sum()*100 })
fig=px.bar(df)    


# %% [markdown]
# 

# %% [markdown]
# ## Findings
# 
# We further conducted EDA analyses on the *UK human trafficking data* and observed that minors had been exploited for organ harvesting for a number of years in UK. 
# 
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-07-05T03:50:37.584625Z","iopub.status.idle":"2023-07-05T03:50:37.585364Z","shell.execute_reply.started":"2023-07-05T03:50:37.585141Z","shell.execute_reply":"2023-07-05T03:50:37.585163Z"}}
# UK trafficking dataset

uk_2013 = pd.read_csv('../input/uk-human-trafficking-data/2013_exploitation_type.csv')
uk_2014 = pd.read_csv('../input/uk-human-trafficking-data/2014_exploitation_type.csv')
uk_2015 = pd.read_csv('../input/uk-human-trafficking-data/2015_exploitation_type.csv')
uk_2016 = pd.read_csv('../input/uk-human-trafficking-data/2016_exploitation_type.csv')
uk_2016.rename( columns={'Claimed Exploitation Type':'Claimed exploitation Type'} , inplace=True)
uk_2016.rename( columns={'Trans- gender':'Transgender'} , inplace=True)
uk_2015.rename( columns={'Transsexual':'Transgender'} , inplace=True)

uk_2013['Year']=2013
uk_2014['Year']=2014
uk_2015['Year']=2015
uk_2016['Year']=2016

df=pd.concat( [uk_2013.loc[:9,:],uk_2014.loc[:10,:],uk_2015.loc[:10,:],uk_2016.loc[:9,:] ] )

px.box( df, y=['Female','Male'], color='Year', x='Claimed exploitation Type' )

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [markdown]
# 
# 
# ## References [MLA] <a type='anchor' id='refs'></a>
# 
# [Back to top](#toc) 
# 
# 1. <a id="ref_stats"></a> Vanessa Bouché and Madeleine Bailey. 2020. The UNODC Global Report on Trafficking in Persons: An Aspirational Tool with Great Potential. The Palgrave International Handbook of Human Trafficking (2020), 163–176.
# 3. <a id="ref_beg"></a> The Borgen Project [Internet] March 7, 2023  https://www.borgenmagazine.com/human-trafficking-in-india/ 
# 4. <a id="ref_fishing"></a> Mileski, Joan P., Cassia Bomer Galvao, and Zaida Denise Forester. "Human trafficking in the commercial fishing industry: a multiple case study analysis." _Marine Policy_ 116 (2020): 103616.
# 5. <a id="ref_lin"></a> Lin Y, Chen P, Ho C. Hotel Recognition to Combat Human Trafficking. Stanford, CA. 2022
# 6. <a id="ref_econ"></a> Athreya, B. ‘Slaves to Technology: Worker control in the surveillance economy’, Anti-Trafficking Review,_ issue 15, 2020, pp. 82-101, https://doi.org/10.14197/atr.201220155
# 7. <a id="ref_harm"></a> Gezie, L.D. and Atinafu, A. (2021), "Physical health symptoms among Ethiopian returnees who were trafficked aboard", International Journal of Migration, Health and Social Care, Vol. 17 No. 2, pp. 215-223. https://doi.org/10.1108/IJMHSC-05-2020-0051
# 8. <a id="ref_tech"></a> Milivojevic S., Moore H., and Segrave M., ‘Freeing the Modern Slaves, One Click at a Time: Theorising human trafficking, modern slavery, and technology’,  Anti-Trafficking Review, issue 14, 2020, pp. 16-32,  https://doi.org/10.14197/atr.201220142
# 9. <a id="ref_defn"></a>  Toney-Butler TJ, Ladd M, Mittel O. Human Trafficking. [Updated 2023 Jan 29]. In: StatPearls [Internet]. Treasure Island (FL): StatPearls Publishing; 2023 Jan-. Available from: https://www.ncbi.nlm.nih.gov/books/NBK430910/
# 10. <a id="ref_analytics"></a> Dimas GL, Konrad RA, Lee Maass K, Trapp AC (2022) Operations research and analytics to combat human trafficking: A systematic review of academic literature. PLoS ONE 17(8): e0273708. https://doi.org/10.1371/journal.pone.0273708
# 11. <a id="ref_freedom"></a> Ian Vásquez, Fred McMahon, Ryan Murphy, and Guillermina Sutter Schneider, _The Human Freedom Index 2022: A Global Measurement of Personal, Civil, and Economic Freedom_ (Washington: Cato Institute and the Fraser Institute, 2022).
# 12. <a id="ref_hotel50k"></a> Stylianou, A., Xuan, H., Shende, M., Brandt, J., Souvenir, R., & Pless, R. (2019). Hotels-50K: A Global Hotel Recognition Dataset. _Proceedings of the AAAI Conference on Artificial Intelligence_, _33_(01), 726-733. https://doi.org/10.1609/aaai.v33i01.3301726 
# 13. <a id="ref_farrell"></a>  Farrell A, Dank M, Ieke de Vries, Kafafian M, Hughes A, Lockwood S. (2019) Failing victims? Challenges of the police response to human trafficking. Criminology and Public Policy https://doi.org/10.1111/1745-9133.124566 
# 14.  <a id="ref_massage"></a> Tobey, Margaret, et al. "Interpretable models for the automated detection of human trafficking in illicit massage businesses." _IISE Transactions_ (2022): 1-14.
# 15. <a id="ref_game"></a> Peterson, Emily. (2019) "Analysis of 2018 Human Trafficking Location Data from skipthegames. com."  Pilot Scholar, University of Portland. Engineering Undergraduate Publications, Presentations and Projects: 9. https://pilotscholars.up.edu/egr_studpubs/9
# 
# 

# %% [markdown]
# # Supplementary <a class="anchor" id="suppl"></a>
# 
# [Back to top](#toc) 
# 
# <img src="https://img.freepik.com/free-vector/human-trafficking-report-infographic-layout-chart_1284-8451.jpg?w=740&t=st=1688421057~exp=1688421657~hmac=e1cde90c0dc1cad10235da5d7dde4346c22b3c359586284fd31b02e352e9a2a1" alt="" style="height: 1000px; width:1000px"/>
# 
# <a href='https://www.dhs.gov/blue-campaign/infographic'>Source: Blue Campaign</a>

# %% [markdown]
# ## Background
# 
# ### Computer science methods and their applications
# 
# As the meta-review Dimas et al. [(2022)](#ref_analytics) did not provide contexts on how these categories of methods were applied in anti-human trafficking, we present them below to facilitate our interpretations of each:
# 
# - **Active learning** refers to a form of machine learning that interactively query a human user (or oracle/ teacher) to label new data samples. It is often used to label large corpuses that are initially unlabelled (example below)
# - **Link inferences** to uncover criminal activities: 
#     - Deceptive (non-sex) recruitment offers found on CraigsList.com, SpaStaff.com, Indeed.com often promote job postings looking for non-sex types of skills
#     - Deceptive recruitment offers and commercial sex sale advertisements that are linked to the same enterprise signifies a potential trafficking organization. 
# - **Simulation** include mathematical modeling techniques that offer *what-if* scenarios to guide the policing-makers; examples models include *discrete-event*, *agent-based* simulations. To detect existing spatiotemporal patterns and predict future behaviour (e.g. target locations), [Keskin et al. 2021](#ref_simulation) used **network simulation** to construct a directed graph that links individual advertisement posts and then perform simulations of random walks on the graph to predict future target locations (of victims).
# 
# #### Prevention 
#  
# Nepal is considered a major "source country"; most of the victims in Nepal are women and children not only suffer gender inequality but extreme poverty. They were often lured with false promises of better jobs, false marriage proposals, or were threatened by indebted families that their children would be sold otherwise. 
# - **Data envelop analysis** (DEA) is done to assess the performance of units within an entity, or across organizations. Identifying which units are relatively more efficient than the rest will help end-users form standardized practices that are most efficient in identifying vulnerable victims.
# - [Dimas et al. 2021](#ref_envelop) use DEA to analyze data collected at the border stations of non-governmental organization to evaluate the performance of an organization called Love Justice International (LJI) in Nepal that is engaged in a trafficking intervention known as *transit-monitoring*, which aims to  identify and intervene potential cases of human trafficking as victims transit through border stations. In a nutshell, their DEA employs *linear programming* to score the efficiencies of stations in such a way that each of the identified inefficient units is outperformed by combination(s) of efficient units.
# 
# 
# #### Tracking 
# 
# Understanding the **supply chain** of trafficking improves coordination of efforts. In doing so, [Ramchandani et al. 2021](#ref_activelearn) extracted 13,568,130 posts from commercial sex websites to study where and how sex workers recruitment occurred. They trained a deep neural network to unmask deceptive posts that businesses promotes for modelling jobs but sells sex, for instance. 
# - Some of the **natural language processing** methods they used:
# 	- Vocabularies such as 'commission', 'audition', 'high pay', 'airfare travel', 'scout' provide domain-specific word contexts; **word2vec** is used to build **word embeddings** of 100 dimensions to encode how frequently pairs of words co-occur
# - To create a training data set, domain-specific human experts had to manually read each unstructured job post, which is a labourious process. Labeling all posts in their corpus was infeasible and labeling a random subset is suboptimal because 99.4% is estimated to be *non-deceptive*. 
# 	- To overcome this challenge, **active learning** is used whereby initial classification of posts provide uncertainty score for each of the preliminary labelings. Their active learning framework then employs two additional metrics to prioritize the labelling of posts from geographically diverse locations and those that promote the discovery of new deception tactics. 
# - They reported that mere use of experts' vocabularies identified three types of recruitment tactics while their framework identified 27 tactics.    
# 
# To identify sources of trafficking, *Seattle Against Slavery* (SAS) has also deployed conversational chatbots disguised as underage females to identify potential trafficked sex buyers. The data collected with these chatbots is used to create buyer profiles. Other pillars of SAS also aim to disrupt transaction of trafficked online sex sales:
# - Potential sex-buyers are identified and redirected to webpages that detail the legal consequences of online sex purchases
# -  Insights on the patterns of buyers and victims are tracked and mined 
# 
# ### Rescues 
# 
# Non-government profits such as the Global Emancipation Network (GEN) have been formed to track victims and identify patterns of their migrations. Other approaches based on computer vision methods include:
# - Identify uniformed children during the first 72 hours of their captures; 
# - **Visual kinship recognition** software are being developed to identify long-time survivors of trafficked children 
# 	- Datasets: 
# 		- **KinFaceW-II** comprised of 200 pairs for 4 types of kinships
# 		- **Families in the Wild** comprised of 11,932 natural family photos from 1,000 families categorized into 11 relationships
#  

# %% [markdown]
#  

# %% [code] {"execution":{"iopub.status.busy":"2023-07-05T03:50:37.586589Z","iopub.status.idle":"2023-07-05T03:50:37.587340Z","shell.execute_reply.started":"2023-07-05T03:50:37.587125Z","shell.execute_reply":"2023-07-05T03:50:37.587145Z"}}
s1 = 'Blue Berries muffin'
pattern = 'Blue Berries'
for match in re.finditer(pattern, s1):    
    s = match.start()
    e = match.end()
    print ('String match "%s" at %d:%d' % (s1[s:e], s, e))

# %% [markdown]
# ## Code to finalize submission

# %% [code] {"execution":{"iopub.status.busy":"2023-07-05T03:50:37.588534Z","iopub.status.idle":"2023-07-05T03:50:37.589219Z","shell.execute_reply.started":"2023-07-05T03:50:37.589018Z","shell.execute_reply":"2023-07-05T03:50:37.589038Z"}}
import pandas as pd
data={}
data['type']=['essay_category', 'essay_url', 'feedback1_url','feedback2_url','feedback3_url'] 
data['value']= ['other', 
                'https://www.kaggle.com/code/', 
                'https://www.kaggle.com/code/',
                'https://www.kaggle.com/code/', 
                'https://www.kaggle.com/code/' ] 
data = pd.DataFrame.from_dict( data )
data = data.set_index('type')
data.to_csv('submission.csv' )
data
