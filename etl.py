import os
import re
import praw
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime, timedelta, timezone

load_dotenv()

def scrape_data():
    '''
    Scrapes the data from r/wallstreet bets for the past 3 hours of posts and outputs it to a csv

    Returns:
        df (pd.Dataframe) - The dataframe containing the uncleaned scraped data from r/wallstreetbets
    '''
    # Set up Reddit API credentials
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="StockRagSystem/1.0"
    )

    # Access r/wallstreetbets subreddit
    subreddit = reddit.subreddit("wallstreetbets")

    # Calculate today's start timestamp
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today = today - timedelta(hours=3)
    today_timestamp = int(today.timestamp())

    posts_data = []
    i = 1
    for post in subreddit.new(limit=500):  # Increased limit to catch more posts
        # Check if post is from the past 3 hours
        if post.created_utc >= today_timestamp:
            print(f'Working on post {i} from {datetime.fromtimestamp(post.created_utc)}')
            posts_data.append({
                "id": post.id,
                "title": post.title,
                "body": post.selftext,
                "upvotes": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "created_utc": datetime.fromtimestamp(post.created_utc),
                "url": post.url
            })

            i += 1
        else:
            # We've reached posts from before today
            print(f'Found older post from {datetime.fromtimestamp(post.created_utc)}')
            break

    # Convert to DataFrame
    df = pd.DataFrame(posts_data)

    return df

def clean_text(text):
    '''
    Helper function to modify columns of a dataframe

    Input:
        text (str) - the uncleaned text data

    Returns:
        text (str) - the cleaned text
    '''
    # Return empty string and empty list if text is not a str
    if not isinstance(text, str):
        return "", []  
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove Reddit formatting (markdown and html formattings)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'&amp;', '&', text)

    # lowercase all text
    text = text.lower()
    
    return text

def clean_df(df):
    '''
    Performs data preprocessing of the dataframe before chunking

    Input:
        - df (pd.Dataframe) - the uncleaned dataframe
    
    Returns:
        - df (pd.Dataframe) - the cleaned dataframe
    '''
    # Remove duplicates
    df = df.drop_duplicates(subset=['id'])

    # Apply cleaning to the body oclumn
    df['body'] = df['body'].apply(clean_text)

    # remove deleted/removed posts from the dataframe
    df = df[~df['body'].str.contains('deleted|removed', case=False)]

    return df

def get_embedding(client, text):
    '''
    Creates the embedding of a given text

    Inputs:
        - client (OpenAI): the OpenAI client to be used for embedding
        - text (str): the text to be embedded
    
    Returns:
        - the embedding of the text
    '''
    # call open ai to embed the text
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    # return the embedding
    return response.data[0].embedding

def create_chunks(df, client):
    '''
    Create the chunks to be uploaded to the vector database. Chunking is done at the 'post' level (i.e. one chunk per reddit post)

    Inputs:
        - df (pd.Dataframe) - The dataframe of reddit posts
        - client (OpenAI) - The OpenAI client to use for creating embeddings
    
    Returns:
        - records (list(dict)) - The list of embeddings and metadata to be uploaded to the vector database
    '''
    records = []
    i = 1
    
    for _, row in df.iterrows():
        print(f'chunking and embedding index {i}')
        if 'title' in row:
            # Format date
            created_date = row['created_utc']
            if isinstance(created_date, datetime):
                formatted_date = created_date.strftime('%Y-%m-%d')
            else:
                formatted_date = str(created_date)
            
            # The formatted text to be embedded
            text_to_embed = f"Date: {formatted_date} Title: {row['title']} Body: {row['body']}"
            
            # the text embedding, id and metadata to be uploaded to the vector db
            records.append({
                "id": row['id'],
                "values": get_embedding(client, text_to_embed),
                "metadata": {
                    "created_utc": row['created_utc'],
                    "title": row['title'],
                    "body": row['body'],
                    "upvotes": row['upvotes']
                }
            })
        
        i += 1

    return records

def delete_old_vectors(index):
    '''
    Deletes all vectors older than 24 hours from Pinecone.

    Inputs:
        - index: The pinecone index to delete vectors from 
    '''
    print("Fetching all metadata from Pinecone...")

    # Fetch all vector metadata
    vector_ids = list(index.list())[0]

    all_vector_ids = index.fetch(ids=vector_ids)
    all_vector_ids = all_vector_ids.vectors

    if not all_vector_ids:
        print("No vectors found to delete.")
        return

    # Get current time and calculate 24-hour threshold
    now = datetime.now(timezone.utc)
    time_threshold = now - timedelta(days=1)

    vectors_to_delete = []

    for vector_id in all_vector_ids:
        # Fetch metadata for each vector
        vector_data = all_vector_ids[vector_id]
        metadata = vector_data["metadata"]

        if 'created_utc' in metadata:
            vector_date = metadata['created_utc']
            vector_date = datetime.fromtimestamp(float(vector_date), tz=timezone.utc)

            # If the vector is older than 24 hours, mark it for deletion
            if vector_date < time_threshold:
                vectors_to_delete.append(vector_id)

    # Delete outdated vectors
    if vectors_to_delete:
        print(f"Deleting {len(vectors_to_delete)} old vectors...")
        index.delete(ids=vectors_to_delete)
        print("Old vectors successfully deleted.")
    else:
        print("No old vectors to delete.")

def run_etl_pipeline():
    '''
    Run the entire ETL pipeline. Extracts data from r/wallstreetbets, cleans the data, chunks it and loads it to be stored in the vector db
    '''
    # create OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI"))

    # create Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE"))

    # scrape the data from reddit
    df = scrape_data()

    # clean the dataframe
    cleaned_df = clean_df(df)

    print('creating embeddings')
    chunks = create_chunks(cleaned_df, openai_client)

    print('finished creating embeddings')

    # Create index if it doesn't exist
    if "wsb-index" not in pc.list_indexes().names():
        print('creating pinecone index')
        pc.create_index(
            name='wsb-index', 
            dimension=1536, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )

    print('uploading records to pinecone')

    # Connect to index
    index = pc.Index("wsb-index")

    # Delete vectors older than 24 hrs from vector db
    delete_old_vectors(index)

    # Upload records
    index.upsert(vectors=chunks)

    print('finished uploading to pinecone')

def main():
    run_etl_pipeline()

if __name__ == '__main__':
    main()