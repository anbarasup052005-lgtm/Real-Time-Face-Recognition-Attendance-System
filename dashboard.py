import streamlit as st
import pandas as pd
import mysql.connector

st.title("📊 Face Attendance Dashboard")

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",
    database="attendance_db"
)

df = pd.read_sql("SELECT * FROM records", conn)

st.dataframe(df)