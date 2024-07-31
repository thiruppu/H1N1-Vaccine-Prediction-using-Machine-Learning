import streamlit as st
import streamlit_option_menu as option
import pandas as pd
from streamlit_extras.let_it_rain import rain
from streamlit_extras.colored_header import colored_header
import plotly.express as px
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
data_for_viz = pd.read_csv("D:\\Data Science\\Project 7\\dashboard_data.csv")

def h1n1_worry_graph():
    fig1 = data_for_viz["h1n1_worry"].value_counts().reset_index()
    fig1.columns = ['h1n1_worry', 'count']
    fig_m_t_count_bar = px.pie(fig1, names='h1n1_worry', values='count', color_discrete_sequence=px.colors.cyclical.HSV)
    fig_m_t_count_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig_m_t_count_bar

def health_worker_graph():
    fig1 = data_for_viz['is_health_worker'].value_counts().reset_index()
    fig1.columns = ['is_health_worker', 'count']
    fig_m_t_count_bar = px.pie(fig1, names='is_health_worker', values='count', color_discrete_sequence=px.colors.cyclical.HSV)
    fig_m_t_count_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig_m_t_count_bar

def sex_graph():
    fig1 = data_for_viz["sex"].value_counts().reset_index()
    fig1.columns = ['sex', 'count']
    fig_m_t_count_bar = px.pie(fig1, names='sex', values='count', color_discrete_sequence=px.colors.cyclical.HSV)
    fig_m_t_count_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig_m_t_count_bar

def h1n1_awarness_graph():
    fig1 = data_for_viz['h1n1_awareness'].value_counts().reset_index()
    fig1.columns = ['h1n1_awareness', 'count']
    fig_m_t_count_bar = px.pie(fig1, names='h1n1_awareness', values='count', color_discrete_sequence=px.colors.cyclical.HSV)
    fig_m_t_count_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig_m_t_count_bar

def routine_graph():
    fig1 = data_for_viz['wash_hands_frequently'].value_counts().reset_index()
    fig1.columns = ['wash_hands_frequently', 'count']
    fig_m_t_count_bar = px.pie(fig1, names='wash_hands_frequently', values='count', color_discrete_sequence=px.colors.cyclical.HSV)
    fig_m_t_count_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig_m_t_count_bar

def race_graph():
    fig1 = data_for_viz['race'].value_counts().reset_index()
    fig1.columns = ['race', 'count']
    fig_m_t_count_bar = px.pie(fig1, names='race', values='count', color_discrete_sequence=px.colors.cyclical.HSV)
    fig_m_t_count_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig_m_t_count_bar

st.set_page_config(layout="wide")

def example():
    rain(
        emoji="💡",
        font_size=54,
        falling_speed=2,
        animation_length="finite",
    )

def example1():
    colored_header(
        label="",
        description="For more details, click the following links to explore:",
        color_name="blue-green-70",
    )

def input_data(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11):

    with open('project07.pkl', 'rb') as file:
        guvi = pickle.load(file)
    input_features = [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11]
    input_tuple = np.array(input_features)
    #print(input_tuple)
    
    input_features_reshaped = np.array(input_tuple).reshape(-1,1)
    #print(input_features_reshaped.shape)
    input_features_reshaped = input_features_reshaped.T
    prediction = guvi.predict(input_features_reshaped)
    #print(prediction[0])
    return prediction[0] 

data_dict = {
    "columns": ["h1n1_worry", "dr_recc_h1n1_vacc", "has_health_insur", "is_h1n1_vacc_effective", "is_h1n1_risky", "sick_from_h1n1_vacc", "is_seas_vacc_effective", "is_seas_risky", "age_bracket", "qualification"],
    "descriptions": [
        "0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried",
        "0=No, 1=Yes",
        "0=No, 1=Yes",
        "0=Zero, 1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective",
        "0=Zero, 1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=Don’t know if it is risky or not, 4=Thinks it is a somewhat high risk, 5=Thinks it is very highly risky",
        "0=Zero, 1=Respondent not worried at all, 2=Respondent is not very worried, 3=Doesn't know, 4=Respondent is somewhat worried, 5=Respondent is very worried",
        "0=Zero, 1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective",
        "0=Zero, 1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=Doesn't know if it is risky or not, 4=Thinks it is somewhat high risk, 5=Thinks it is very highly risky",
        "0=18 - 34 Years, 1=35 - 44 Years, 2=45 - 54 Years, 3=55 - 64 Years, 4=64+ Years",
        "0=12 Years, 1=<12 Years, 2=College Graduate, 3=Some College, 4=Zero"
    ]
}


st.markdown(
    """
    <style>
    .stApp {
        background-color:#2F3C7E;
        color: #FBEAEB;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp p, .stApp div {
        color: #FBEAEB;
    }
    .header {
        align:center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    menu = option.option_menu(
        "Main Menu", ["Overview", "Graphs", "Prediction"],
        icons=['house', 'bar-chart', 'clipboard-check'],
        menu_icon="cast",
        default_index=0
    )

if menu == "Overview":
    st.title("Vaccine Usage Analysis and Prediction 💉")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" ")
        st.header("**Skills Takeaway:**")
        st.subheader("🐍 Python scripting | 🐼 Pandas | 📈 Data Visualization | 🦾 Machine Learning")
        st.write(" ")
        st.header("**Description:**")
        st.write("""
        This project aims to predict the likelihood of individuals taking the H1N1 flu vaccine using Logistic Regression by analyzing a dataset containing various features related to individuals' behaviors, 
        perceptions, and demographics. The dataset includes information such as age, gender, education level, health status, and preventive behaviors like handwashing and mask-wearing. 
        The data preprocessing steps involve handling missing values, encoding categorical variables, and normalizing numerical features to prepare the data for accurate modeling. 
        Exploratory Data Analysis (EDA) is performed to uncover patterns and relationships within the data, providing insights into key factors influencing vaccine acceptance.
        
        By building a predictive model, we can estimate the probability of vaccine acceptance for different individuals based on their characteristics and attitudes. 
        This model aids healthcare professionals and policymakers in effectively targeting vaccination campaigns, ensuring that efforts are directed towards populations that are less likely to receive the vaccine and require more encouragement. 
        The ultimate goal is to enhance public health strategies, improve vaccination rates, and better manage the spread of the H1N1 flu by understanding and addressing the factors that influence individuals' decisions to get vaccinated.
        """)
    
    with col2:
        st.image('vaccine_.png')
    
    example1()
    
    col1_1, col1_2, col1_3 = st.columns(3)
    with col1_1:
        st.image("linkedin.png")
        st.write("[LinkedIn](https://www.linkedin.com/in/thiruppugazhan-s-277705282/)")
    with col1_2:
        st.image("instagram.png")
        st.write("[Instagram](https://instagram.com/_thiruppugazhan)")
    with col1_3:
        st.image("github.png")
        st.write("[GitHub](https://github.com/thiruppu)")

if menu == "Graphs":
    st.title("Graphical Representation of User Analysis")
    col1_1, col1_2, col1_3 = st.columns(3)
    
    with col1_2:
        options = st.selectbox('Select', ["h1n1 worry", "health worker", "sex", "h1n1 awarness", "race", "routine graph"])
    
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        if options == "h1n1 worry":
            st.subheader("H1N1 Worry:")
            st.image("coronavirus.png")
                
    with col2_2:
        if options == "h1n1 worry":
            chart_h1n1_worry = h1n1_worry_graph()
            st.plotly_chart(chart_h1n1_worry)
            st.write("""
            This graph shows the count of responses for each category of H1N1 worry. 
            The categories include "Not worried at all," "Not very worried," "Somewhat worried," and "Very worried." 
            This visualization provides a clear depiction of how many individuals fall into each category, 
            helping us understand the distribution of worry levels within the population. 
            By analyzing this graph, we can gain insights into the general sentiment and anxiety surrounding the H1N1 flu, 
            which can be crucial for tailoring public health messages and interventions.
            """)
            
    with col2_1:
        if options == "health worker":
            st.subheader("Health Worker:")
            chart_health_worker = health_worker_graph()
            st.plotly_chart(chart_health_worker)
            st.write("""
            The graph displays the distribution of individuals based on their status as health workers.This categorical variable is divided into two groups: those who are health workers and those who are not. 
            By visualizing this data, we can understand the proportion of health workers within the dataset,which can be crucial for analyzing their behavior and attitudes towards health-related issues, such as vaccination. 
            This information helps in identifying any significant differences in vaccine acceptance and other health-related behaviors between health workers and the general population,
            providing insights that can inform targeted public health strategies and interventions.
            """)
        
    with col2_2:
        if options == "health worker":
            st.image("hw.png")
    
    with col2_1:
        if options == "sex":
            st.subheader("Sex Ratio:")
            chart_sex = sex_graph()
            st.plotly_chart(chart_sex)
            st.write("""
            The graph illustrates the distribution of individuals based on their sex, 
            categorizing them into male and female groups. This visualization helps in understanding the sex ratio within the dataset, 
            providing a clear depiction of the proportion of males and females. By analyzing this data, we can identify any potential gender-based disparities in various behaviors, perceptions, and attitudes towards health-related issues, including vaccination. This analysis is crucial for recognizing patterns and trends that may exist between different sexes, 
            which can inform more tailored and effective public health strategies.
            """)
        
    with col2_2:
        if options == "sex":
            st.image("sex.png")

    with col2_1:
        if options == "h1n1 awarness":
            st.subheader("H1N1 Awarness:")
            chart_h1n1_awarness = h1n1_awarness_graph()
            st.plotly_chart(chart_h1n1_awarness)
            st.write("""
            Awareness levels play a crucial role in public health, particularly in the context of disease prevention and vaccination.
            Higher awareness is often associated with better preventive behaviors and higher vaccine acceptance rates. 
            By examining the distribution of H1N1 awareness, healthcare professionals and policymakers can tailor their communication strategies to effectively reach and educate those with lower awareness levels. 
            This targeted approach can help in improving overall public health preparedness and response,
            ensuring that more individuals are informed and equipped to take appropriate actions to protect themselves and their communities from the H1N1 flu.
            """)
        
    with col2_2:
        if options == "h1n1 awarness":
            st.image("vaccine02.png")
    
    with col2_1:
        if options == "race":
            st.subheader("Race:")
            chart_race = race_graph()
            st.plotly_chart(chart_race)
            st.write("""
            Analyzing racial distribution is important for identifying any potential disparities or patterns in health-related behaviors and outcomes.
            Understanding how different racial groups are represented can help tailor public health initiatives and interventions to address specific needs and concerns within each group. 
            This approach ensures that health strategies are inclusive and equitable, 
            promoting better health outcomes across diverse populations and addressing any potential health inequities that may exist.
            """)
        
    with col2_2:
        if options == "race":
            st.image("race.png")

    with col2_1:
        if options == "routine graph":
            st.subheader("Wash hands frequently:")
            st.image("fff.png")

        
    with col2_2:
        if options == "routine graph":
            
            chart_routine = routine_graph()
            st.plotly_chart(chart_routine)
            st.write("""
            The graph presents the frequency with which individuals wash their hands, categorized into different levels of frequency. 
            This visualization provides insights into how often people engage in this essential preventive behavior, 
            ranging from those who wash their hands frequently to those who do so less often. By examining this data,
            we can assess the general adherence to hand hygiene practices within the population.
            """)        


if menu == "Prediction":
    st.title("Machine Learning Prediction 🦾")
    st.subheader("Please enter the following details:")
    #example()
    col1_1, col1_2, col1_3, col1_4 = st.columns(4)

    with col1_2:
        var1 = st.number_input("Unique ID", key="unique_id",placeholder="1-100",min_value=0)
        var2 = st.selectbox("H1N1 Worry (0-3)", options=[0, 1, 2, 3])
        var3 = st.selectbox("Doctor Recommended H1N1 Vaccine", options=[0, 1])
        var4 = st.selectbox("Has Health Insurance", options=[0, 1])
        var5 = st.selectbox("Is H1N1 Vaccine Effective", options=[0, 1, 2, 3, 4, 5])
        var6 = st.selectbox("Is H1N1 Risky", options=[0, 1, 2, 3, 4])

    with col1_3:
        var7 = st.selectbox("Sick from H1N1 Vaccine", options=[0, 1, 2, 3, 4, 5])
        var8 = st.selectbox("Is Seasonal Vaccine Effective", options=[0, 1, 2, 3, 4, 5])
        var9 = st.selectbox("Is Seasonal Risky", options=[0, 1, 2, 3, 4, 5])
        var10 = st.selectbox("Age Bracket", options=[0, 1, 2, 3])
        var11 = st.selectbox("Qualification", options=[0, 1, 2, 3, 4])
        
    with col1_3:
        st.write(" ")
        st.write(" ")
        prediction_button = st.button("Predict")
    
    
    if prediction_button:
        prediction = input_data(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11)
        #st.subheader(prediction)
        if prediction is not None:
            st.header(f"The predicted likelihood of taking the H1N1 flu vaccine is: {'Yes (val=1)' if prediction == 1 else 'No (val=0)'}")

    col3_1,col3_2,col3_3 = st.columns(3)
    data_dict = {
        "columns": ["h1n1_worry", "dr_recc_h1n1_vacc", "has_health_insur", "is_h1n1_vacc_effective", "is_h1n1_risky", "sick_from_h1n1_vacc", "is_seas_vacc_effective", "is_seas_risky", "age_bracket", "qualification"],
        "descriptions": [
            "0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried",
            "0=No, 1=Yes",
            "0=No, 1=Yes",
            "0=Zero, 1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective",
            "0=Zero, 1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=Don’t know if it is risky or not, 4=Thinks it is a somewhat high risk, 5=Thinks it is very highly risky",
            "0=Zero, 1=Respondent not worried at all, 2=Respondent is not very worried, 3=Doesn't know, 4=Respondent is somewhat worried, 5=Respondent is very worried",
            "0=Zero, 1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective",
            "0=Zero, 1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=Doesn't know if it is risky or not, 4=Thinks it is somewhat high risk, 5=Thinks it is very highly risky",
            "0=18 - 34 Years, 1=35 - 44 Years, 2=45 - 54 Years, 3=55 - 64 Years, 4=64+ Years",
            "0=12 Years, 1=<12 Years, 2=College Graduate, 3=Some College, 4=Zero"
        ]
    }
    final_data = pd.DataFrame(data_dict)
    
    with col3_2:
        st.dataframe(final_data)
        
