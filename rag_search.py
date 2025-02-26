import numpy as np
from scipy.spatial.distance import cosine
from etl import get_embedding
from datetime import datetime
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

def get_gpt_prompt(stock_portfolio='*currently your stock portfolio is empty*'):
    '''
    Creates the system prompt for GPT to use

    Inputs:
        - stock_portfolio (str) - The stocks in a user's portfolio
    
    Returns
        - (str) - The system prompt to pass into the selected LLM model
    '''

    return f"""
    You are a stock market analyst who specializes in analyzing the market. 
    The following prompt you will receive will be from r/WallStreetBets discussions.

    ------
    Here is the current stock portfolio you have available at your disposal:

    {stock_portfolio}
    ------

    Absolute Rules:
        - **DO NOT SELL any stock that is not explicitly in the portfolio above.** If a stock is not in the portfolio, you **must ignore it.**
        - If you attempt to recommend a sale for a stock **not in the portfolio**, the response is INVALID.
        - **If a stock is not in the portfolio, it cannot be in the Sell or Hold list.**
        - You must take in consideration the posts, but you do not need to base your decisions off of just the posts alone. 
        - You have access to the entire NYSE and NASDAQ to give recommendations. Do not forget to take that into consideration
        - You should respond by listing what you would buy (any stock) and sell IF THE STOCK IS IN YOUR PORTFOLIO
        - YOU MAY NOT just say short-term. You must give a timeframe for how long they should hold the stock. You may conduct day trades if you think it would be the most beneficial.
        - The Format of your response MUST BE ONLY:
            'Buy:

            1. [ticker 1]: how long they should hold it (sell by EOD)
            2. [ticker 2]: how long they should hold it (3-6 months)
            3. [ticker 3]: how long they should hold it (1+ years)
            ...

            Sell:

            1. [ticker 4]
            2. [ticker 5]
            ...

            Hold:

            1. [ticker 6]
            2. [ticker 7]
            ...

            Insights:

            1. [ticker 1]: insight
            2. [ticker 2]: insight
            ...'

        - **IMPORTANT**:
            - If no stocks in the portfolio should be sold, say: **"No stocks from the portfolio should be sold at this time."**
            - If no stocks should be held, say: **"No stocks in the portfolio are suitable for holding at this time."**
            - If a stock is meant to be held for short term you must not say give a specific time-frame not just 'short-term'
            - Any violation of these rules means your response is incorrect.
    """

def cosine_similarity(vec1, vec2):
    '''
    Computes the cosine similarity between two vectors

    Inputs:
        - vec1 - A numerical vector
        - vec2 - A numerical vector
    
    Returns:
        - The cosine similarity normalized to be >= 0
    '''
    return 1 - cosine(vec1, vec2)

def get_todays_stock_insights(pinecone_index, openai_client):
    '''
    Creates the user prompt (i.e. query) to be passed into the specified LLM

    Inputs:
        - pinecone_index: the database where the posts are stored
    
    Returns:
        - prompt (str): The query to feed into the LLM with the RAG system's results included in the prompt
    '''
    # get the date to use in the query for cosine similarity
    today = datetime.now().strftime('%Y-%m-%d')

    # create the base query with keywords to calculate cosine similarity
    query = f"Date: {today} stocks trading buy sell hold"

    # embed the query prompt
    query_vector = get_embedding(openai_client, query)
    
    # get all vectors from the database to calculate cosine similarity
    vector_ids = list(pinecone_index.list())[0]

    all_vectors = pinecone_index.fetch(ids=vector_ids)
    all_vectors = all_vectors.vectors
    
    # calculate cosine similarity between our query vector and all vectors in our database
    results = []
    for id in all_vectors:
        vector_data = all_vectors[id]
        similarity = 1 - cosine(query_vector, vector_data["values"])
        results.append((id, similarity, vector_data["metadata"]))
    
    # get the top 10 results from our calculated cosine similarity
    results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    
    # format each of the found posts in the format we want to put into our LLM
    context = "\n\n".join([
        f"----------------------------\n"
        f"Post {i+1}\n"
        f"Title: {r[2]['title']}\n"
        f"Body: {r[2]['body']}\n"
        f"Date: {r[2]['created_utc']}\n"
        f"Upvotes: {r[2]['upvotes']}\n"
        f"----------------------------\n"
        for i, r in enumerate(results)
    ])
    
    # create the user prompt for the LLM
    prompt = f"""What stocks should I buy or sell today: ({today}) given your insight and the following posts as context

    Top posts:
    {context}

    Analyze these posts and recommend the top 3 potential trades.
    """
    
    return prompt

def get_stock_predictions(portfolio):
    '''
    Feed our data into the LLM to get a response

    Inputs:
        - portfolio (str) - The stocks in a user's portfolio
    
    Returns:
        - (str) - The response from the specified LLM
    '''
    # create Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE"))

    # create OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI"))

    # create default response if portfolio is empty
    if portfolio == '':
        portfolio = '*currently your stock portfolio is empty*'

    # get GPT system prompt
    gpt_prompt = get_gpt_prompt(portfolio)
    
    # connect to the pinecone index
    index = pc.Index("wsb-index")

    # create query, calculate cosine similarity and get user prompt to feed into the LLM
    query_results = get_todays_stock_insights(index, client)

    # generate response from GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": gpt_prompt},
            {"role": "user", "content": query_results}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    # return response from GPT
    return response.choices[0].message.content