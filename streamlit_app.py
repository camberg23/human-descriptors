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
openai.api_key = st.secrets["OPENAI_KEY"]

# Helper Functions
# Function to compute embeddings for a given text
def get_embeddings(text):
    response = openai.Embedding.create(
        input=f'a {text} person',
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

def save_embeddings(embeddings_dict, filename):
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print(f"Embeddings saved to {filename}.pickle")

def load_embeddings(filename):
    with open(filename + '.pickle', 'rb') as handle:
        embeddings_dict = pickle.load(handle)
#     print(f"Embeddings loaded from {filename}.pickle")
    return embeddings_dict

def capture_frames_from_plotly(fig, angles, directory="frames", zoom_factor=1):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    frame_paths = []
    for angle in angles:
        fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=1),  # Keeping the "up" direction fixed to the z-axis
                                            center=dict(x=0, y=0, z=0),
                                            eye=dict(x=zoom_factor * np.cos(np.radians(angle)),
                                                     y=zoom_factor * np.sin(np.radians(angle)),
                                                     z=0)))  # Setting z to 0 to get the straight-on view in z dimension
        frame_path = os.path.join(directory, f"frame_{angle}.png")
        fig.write_image(frame_path)
        frame_paths.append(frame_path)
    
    return frame_paths

def create_gif_from_frames(frame_paths, gif_path="rotating_graph.gif", duration=150):
    """
    Create a GIF from the provided frame paths.
    
    Args:
    - frame_paths (list): List of paths to individual frames/images.
    - gif_path (str): Path where the GIF will be saved.
    - duration (int): Duration each frame is displayed in milliseconds. Default is 100ms.
    
    Returns:
    - None
    """
    with imageio_v2.get_writer(gif_path, mode='I', duration=duration, loop=0) as writer:
        for frame_path in frame_paths:
            image = imageio_v2.imread(frame_path)
            writer.append_data(image)

def is_valid_word(word):
    # Check word using the Datamuse API
    response = requests.get(f"https://api.datamuse.com/words?sp={word}&max=1")
    return len(response.json()) > 0

def load_sentiment_dict(path):
    """Load the sentiment dictionary from a given path."""
    with open(path, "rb") as f:
        return pickle.load(f)

def get_N_closest_words(descriptor, sentiment_dict, N=5):
    """Get N closest words with a similar sentiment score within the same sentiment label."""
    label, score = sentiment_dict[descriptor]["label"], sentiment_dict[descriptor]["score"]
    
    # Filter the words with the same sentiment label and exclude the provided descriptor
    same_label_words = {word: info["score"] for word, info in sentiment_dict.items() 
                        if info["label"] == label and word != descriptor}
    
    # Sort the words by the difference in sentiment scores
    sorted_words = sorted(same_label_words.keys(), key=lambda word: abs(same_label_words[word] - score))
    
    # Return the top N closest words
    return sorted_words[:N]

# Function to load the RoBERTa model for sentiment analysis
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_task

def predict_sentiment(descriptor, N=5):
    """
    Predict the sentiment of a given descriptor using the sentiment dictionary and the RoBERTa model.
    
    Args:
    - descriptor (str): The human descriptor for which sentiment is to be predicted.
    - sentiment_dict (dict): Dictionary of descriptors with their sentiment label and score.
    - N (int): Number of closest words to return.
    
    Returns:
    - str, float: Sentiment label and score.
    """
    sentiment_dict = load_sentiment_dict('sentiment_dict.pkl')
    # If the descriptor is in the sentiment dictionary, use the pre-computed values
    if descriptor in sentiment_dict:
        label = sentiment_dict[descriptor]["label"]
        score = sentiment_dict[descriptor]["score"]
    else:
        # Ensure the descriptor is a valid word
        if not is_valid_word(descriptor):
            raise ValueError(f"'{descriptor}' is not a recognized word.")
        
        # Predict sentiment using the pre-trained RoBERTa model
        sentiment_task = load_sentiment_model()
        output = sentiment_task(f'a {descriptor} person')
        label = output[0]['label'].upper()
        score = output[0]['score']
        
        # Add the new descriptor to the sentiment dictionary
        sentiment_dict[descriptor] = {"label": label, "score": score}
    
    # Get N closest words with similar sentiment
    closest_words = get_N_closest_words(descriptor, sentiment_dict, N)
    
    result1 = f"{descriptor.capitalize()} is considered to have a {label} connotation, {round(score*100,1)}% match."
    result2 = f"{N} most connotatively similar descriptors: {', '.join(closest_words)}"
    
    return label, score, result1, result2

# Inner functions
def reduce_dimensions_to_3D(embeddings):
    pca = PCA(n_components=3)
    transformed_embeddings = pca.fit_transform(embeddings)
    return transformed_embeddings, pca

def interpret_clusters(embeddings_dict, centroid, n_closest):
    word_centroid_list = [(word, embedding) for word, embedding in embeddings_dict.items()]
    word_centroid_list.sort(key=lambda x: euclidean_distances([centroid], [x[1]]))
    closest_words = [x[0] for x in word_centroid_list[:n_closest]]
    return closest_words

def get_clusters(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=23)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

def get_cluster_of_descriptor(descriptor, embeddings_dict, n_clusters):
    descriptor_embedding = embeddings_dict[descriptor]
    embeddings = list(embeddings_dict.values())
    
    # Find the index of the descriptor embedding in the embeddings list
    descriptor_index = next(i for i, emb in enumerate(embeddings) if np.array_equal(emb, descriptor_embedding))
    
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    # Return the cluster ID of the specified descriptor
    return labels[descriptor_index]

def get_centroid_of_cluster(embeddings_dict, cluster_id, n_clusters=13):
    _, kmeans = get_clusters(list(embeddings_dict.values()), n_clusters)
    return kmeans.cluster_centers_[cluster_id]

def get_similar_descriptors(descriptor, embeddings_dict, N=5):
    """Get N descriptors that are most similar to the input descriptor."""
    # Check if the descriptor exists in the embeddings_dict
    if descriptor not in embeddings_dict:
        raise ValueError(f"{descriptor} not found in the embeddings dictionary.")
        
    # Calculate the cosine similarity between the descriptor and all other words
    descriptor_embedding = embeddings_dict[descriptor]
    cosine_similarities = {word: np.dot(descriptor_embedding, word_embedding) / (np.linalg.norm(descriptor_embedding) * np.linalg.norm(word_embedding))
                           for word, word_embedding in embeddings_dict.items()}
    
    # Sort the words by similarity
    sorted_words = sorted(cosine_similarities.keys(), key=lambda word: cosine_similarities[word], reverse=True)
    
    # Remove the input descriptor from the list
    sorted_words = [word for word in sorted_words if word != descriptor]
    
    # Return the top N similar words
    return sorted_words[:N]

def visualize_embeddings_complete(embeddings_dict, n_clusters, n_words, highlight_word=None, gif=False):
    """
    Visualize the embeddings in a 3D cluster space.
    
    Args:
    - embeddings_dict (dict): Dictionary containing words and their embeddings.
    - n_clusters (int): Number of clusters for KMeans.
    - n_words (int): Number of closest words to the centroid to display.
    - highlight_word (str, optional): Word to highlight in the visualization.
    
    Returns:
    - fig (plotly.graph_objects.Figure): The 3D visualization.
    - frame_paths (list): Paths to the frames captured for gif creation.
    """

    # Begin main function
    labels, kmeans = get_clusters(list(embeddings_dict.values()), n_clusters)
    transformed_embeddings, pca = reduce_dimensions_to_3D(list(embeddings_dict.values()))
    
    word_cluster_map = dict(zip(embeddings_dict.keys(), labels))
    
    df = pd.DataFrame(transformed_embeddings, columns=['x', 'y', 'z'])
    df['label'] = labels
    df['adjective'] = embeddings_dict.keys()
    
    color_list = list(px.colors.qualitative.Plotly)
    colors = [color_list[label % len(color_list)] for label in df['label']]
    
    sizes = [15 if word == highlight_word else 5 for word in df['adjective']]
    highlight_colors = ['red' if word == highlight_word else color for word, color in zip(df['adjective'], colors)]
    
    fig = go.Figure()
    
    scatter = go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='markers',
                           marker=dict(color=highlight_colors, size=sizes),
                           text=df['adjective'], hoverinfo='text', showlegend=False)
    fig.add_trace(scatter)
    
    transformed_centroids = pca.transform(kmeans.cluster_centers_)
    
    # Place user's word first in the legend
    if highlight_word:
        fig.add_trace(go.Scatter3d(x=[df[df['adjective'] == highlight_word]['x'].values[0]], 
                                   y=[df[df['adjective'] == highlight_word]['y'].values[0]], 
                                   z=[df[df['adjective'] == highlight_word]['z'].values[0]], 
                                   mode='markers', 
                                   marker=dict(size=15, color='red', symbol='circle', line=dict(color='Black', width=1)),
                                   showlegend=True, name=f"Selected word: {highlight_word}"))
    cluster_texts = []
    
    for i, centroid in enumerate(kmeans.cluster_centers_):
        closest_words = interpret_clusters(embeddings_dict, centroid, 8)
        legend_text = f"Cluster {i+1}: {', '.join(closest_words)}"
        cluster_texts.append(legend_text)
        fig.add_trace(go.Scatter3d(x=[transformed_centroids[i][0]], y=[transformed_centroids[i][1]], z=[transformed_centroids[i][2]], mode='markers',
                                   marker=dict(size=8, color=color_list[i % len(color_list)], symbol='diamond',
                                               line=dict(color='Black', width=1)),
                                   showlegend=True, name=legend_text))

    
    fig.update_layout(title_text=f"{n_clusters} Clusters of Human Descriptors",
                  title_x=0.25, title_y=0.92, title_font_size=24, # Add title_y attribute
                  scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                  autosize=False, width=1200, height=1000, 
                  legend=dict(y=-0.1, x=0.5, xanchor='center', orientation='h'))

    
    if gif:
        angles = list(range(0, 360, 5))
        frame_paths = capture_frames_from_plotly(fig, angles)

        # Generate the gif
        create_gif_from_frames(frame_paths)

    return fig, word_cluster_map, cluster_texts

def get_similar_descriptors(descriptor, embeddings_dict, N=5):
    """Get N descriptors that are most similar to the input descriptor."""
    # Check if the descriptor exists in the embeddings_dict
    if descriptor not in embeddings_dict:
        raise ValueError(f"{descriptor} not found in the embeddings dictionary.")
        
    # Calculate the cosine similarity between the descriptor and all other words
    descriptor_embedding = embeddings_dict[descriptor]
    cosine_similarities = {word: np.dot(descriptor_embedding, word_embedding) / (np.linalg.norm(descriptor_embedding) * np.linalg.norm(word_embedding))
                           for word, word_embedding in embeddings_dict.items()}
    
    # Sort the words by similarity
    sorted_words = sorted(cosine_similarities.keys(), key=lambda word: cosine_similarities[word], reverse=True)
    
    # Remove the input descriptor from the list
    sorted_words = [word for word in sorted_words if word != descriptor]
    
    # Return the top N similar words
    return sorted_words[:N]

# Main Function for Analyze Descriptor

def analyze_descriptor_text(descriptor, n_clusters=13, n_words=15):
    """
    Analyze a given descriptor:
    - Identify and print descriptors in its cluster.
    - Identify and print similar descriptors.
    - Identify and print opposite descriptors.
    """
    results = []
    sentiment_dict = load_sentiment_dict('sentiment_dict.pkl')
    embeddings_dict = load_embeddings('condon_cleaned')
    
    if descriptor not in embeddings_dict:
        if not is_valid_word(descriptor):
            raise ValueError(f"'{descriptor}' is not a recognized word.")
        embeddings_dict[descriptor] = get_embeddings(descriptor)

    # Identifying the cluster of the descriptor
    cluster_id = get_cluster_of_descriptor(descriptor, embeddings_dict, n_clusters)
    centroid = get_centroid_of_cluster(embeddings_dict, cluster_id, n_clusters=n_clusters)
    closest_words_to_centroid = interpret_clusters(embeddings_dict, centroid, n_words)
    # results.append(f"'{descriptor.capitalize()}' belongs to cluster {cluster_id} of {n_clusters}: {', '.join(closest_words_to_centroid)}")

    # Identifying descriptors most similar and opposite to the input word
    similar = get_similar_descriptors(descriptor, embeddings_dict, N=n_words)
    results.append(f"Descriptors most mathematically similar to '{descriptor}': {', '.join(similar)}")

    # Sentiment results
    __, __, r1, r2 = predict_sentiment(descriptor)
    results.append(r1)
    results.append(r2)

    results.append(f"'{descriptor.capitalize()}' belongs to cluster {cluster_id} of {n_clusters}: {', '.join(closest_words_to_centroid)}")
    results.append("(Visualize all of the clusters by clicking the button below!)")
    return results

def analyze_descriptor_visual(descriptor, n_clusters=13, n_words=15, gif=False):
    """
    Visualize the descriptor in a 3D cluster space.
    """
    embeddings_dict = load_embeddings('condon_cleaned')
    if descriptor not in embeddings_dict:
        if not is_valid_word(descriptor):
            raise ValueError(f"'{descriptor}' is not a recognized word.")
        embeddings_dict[descriptor] = get_embeddings(descriptor)

    fig, word_cluster_map, cluster_text = visualize_embeddings_complete(embeddings_dict, n_clusters, 
                                          n_words, highlight_word=descriptor, gif=gif)
    return fig

# Main Function for Descriptor Blender
def descriptor_blender(descriptors, N=10):
    """
    Combines a list of descriptors using additive method and finds 
    the words in the embeddings_dict that are closest to this combined representation without clustering.
    
    Args:
    - descriptors (list of str): List of descriptors.
    - embeddings_dict (dict): Dictionary of descriptors with their embeddings.
    - N (int): Number of closest descriptors to return. Default is 10.
    
    Returns:
    - None: Prints the descriptors that are close to the combined representation of the input descriptors.
    """
    # Check and compute embeddings for missing descriptors
    embeddings_dict = load_embeddings('condon_cleaned')
    descriptors = cleaned_descriptors = [desc.strip().lower() for desc in descriptors]
    intersection_words = []
    
    for descriptor in descriptors:
        if descriptor not in embeddings_dict:
            if is_valid_word(descriptor):
                embedding = get_embeddings(descriptor)
                embeddings_dict[descriptor] = embedding
            else:
                raise ValueError(f"{descriptor} not found in the embeddings dictionary, and it's not a valid descriptor.")
    
    # Compute the combined embedding using the additive method
    combined_embedding = sum([embeddings_dict[descriptor] for descriptor in descriptors])
    
    # Remove the original descriptors to not have them in the result
    words_to_compare = {word: embedding for word, embedding in embeddings_dict.items() if word not in descriptors}
    
    # Calculate enriched scores for words
    distances_to_combined = {word: np.linalg.norm(embedding - combined_embedding) for word, embedding in words_to_compare.items()}
    enrich_scores = {word: distance * sum([np.linalg.norm(embeddings_dict[descriptor] - embeddings_dict[word]) for descriptor in descriptors]) for word, distance in distances_to_combined.items()}
    
    # Get the top N words based on enriched scores
    closest_words = sorted(enrich_scores.keys(), key=lambda word: enrich_scores[word])[:N]
    return closest_words
    
# Streamlit App
st.title("Human Descriptor Analyzer & Blender")
st.write("Analyze descriptors from Condon adjective dataset (used to create the Big Five), and blend them to find interesting intersections.")

st.header("Analyze Descriptor")
descriptor = st.text_input("Enter any adjective that describes human personality:")
descriptor = descriptor.replace(" ", "").replace(",", "")
n_clusters = st.number_input("Number of Clusters:", min_value=1, value=25, step=1)
n_similar = st.number_input("Number of similar words to return:", min_value=1, value=15, step=1)

# Placeholders for analyze button, results, visualize button, and visualization
analyze_button_placeholder = st.empty()
analysis_results_placeholder = st.empty()
visualize_button_placeholder = st.empty()
visualization_placeholder = st.empty()

# When the "Analyze" button is pressed, only textual insights will be shown
if analyze_button_placeholder.button("Analyze"):
    if " " in descriptor or "," in descriptor or len(descriptor) != 1:
        st.warning("Please enter a single adjective without spaces or commas.")
    with st.spinner('Analyzing the descriptor...'):
        results = analyze_descriptor_text(descriptor, n_clusters, n_similar)
        st.session_state['analysis_results'] = results

# New button for visualization
if visualize_button_placeholder.button("Visualize your descriptor in full 3D space (interactive)"):
    with st.spinner('Generating the 3D clustering visualization...this will take about 10 seconds.'):
        plot = analyze_descriptor_visual(descriptor, n_clusters, n_similar)
        st.session_state['visualization'] = plot

# Display stored results and visualization in their respective placeholders
if 'analysis_results' in st.session_state:
    all_results = "\\n\\n".join(st.session_state['analysis_results'])
    analysis_results_placeholder.markdown(all_results)

if 'visualization' in st.session_state:
    visualization_placeholder.plotly_chart(st.session_state['visualization'], use_container_width=True)

# Remaining Streamlit code related to the Descriptor Blender
st.header("Descriptor Blender")
words_to_blend = st.text_area("Enter words to blend (comma-separated):").split(',')
num_output_words = st.number_input("Number of output words:", min_value=1, value=20, step=1)

if st.button("Blend"):
    if len(words_to_blend) < 2:
        st.warning("Please enter at least two descriptors to blend.")
    intersection_words = descriptor_blender(words_to_blend, num_output_words)
    st.write("Descriptors that blend your input:")
    
    # Create two columns to display the words
    col1, col2 = st.columns(2)
    
    # Split the words into two lists
    half_len = len(intersection_words) // 2
    words_col1 = intersection_words[:half_len]
    words_col2 = intersection_words[half_len:]
    
    for word in words_col1:
        col1.write(word)
    
    for word in words_col2:
        col2.write(word)
