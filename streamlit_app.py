# Necessary Imports
import streamlit as st
# Standard libraries
import time
import os
import base64

# Data manipulation and analysis
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# Plotting and visualization
import plotly.graph_objects as go
import plotly.express as px
import imageio.v2 as imageio_v2

# Display utilities
from IPython.display import display, HTML

# External libraries
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, logging
logging.set_verbosity_error()

# Serialization
import pickle

# OpenAI
import openai
# Set up OpenAI API key
openai_key = st.secrets["OPENAI_KEY"]

st.write("Hello, Streamlit!")
