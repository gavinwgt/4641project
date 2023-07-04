from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json
import numpy as np
from lyricsgenius import Genius
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

# print(client_id, client_secret)

def get_token():
    auth_str = client_id + ":" + client_secret
    auth_bytes = auth_str.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type":  "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

def get_auth_headers(token):
    return {"Authorization": "Bearer " + token}

def search_for_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_headers(token)
    query = f"?q={artist_name}&type=artist&limit=1"

    query_url = url + query
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)["artists"]["items"]
    
    if len(json_result) == 0:
        print("no artist with this name exists...")
        return None
    return json_result[0]

def get_songs_by_artist(token, artist_id):
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=US"
    headers = get_auth_headers(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]
    return json_result

def get_top_50(token):
    url = f"https://api.spotify.com/v1/playlists/6UeSakyzhiEt4NB3UAd6NQ"
    headers = get_auth_headers(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]["items"]
    names = []
    ids = []
    for i in json_result:
        names.append(i["track"]["name"])
        ids.append((i["track"]["id"], i["track"]["artists"][0]['name']))
    res = {names[j]: ids[j] for j in range(len(names))}
    return res

def get_feats(token, song_id):
    url = f"https://api.spotify.com/v1/audio-features/{song_id}"
    headers = get_auth_headers(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result

def make_data(token):
    unorg_data = get_top_50(token)
    numarr = []
    numarrname = []
    for i,j in zip(list(unorg_data.values()), list(unorg_data.keys())):
        temp = get_feats(token,i[0])
        print(temp)
        numarr.append(list(temp.values())[:11] + list(temp.values())[len(list(temp.values())) - 2:])
        numarrname.append([j])
    # print(np.array(numarrname))
    # return np.array(numarrname)
    return np.array(numarr)

def lyrics(name_and_artist):
    genius = Genius("rDmBG-hI4l1PXuGxpX163DkFcRTy_xmFkg5SjZlJNBRyNquMKzrc5NXQuuumjVuE")
    key = list(name_and_artist.keys())
    val = list(name_and_artist.values())
    for i in range(len(key)):
        song = genius.search_song(key[i], val[i][1])
        f = open(key[i] + ".txt", "w", encoding="utf-8")
        f.write(song.to_dict()['lyrics'])
    # song = genius.search_song('Self Love', 'Metro Boomin')
    # f = open('Self Love' + '.txt', 'w', encoding="utf-8")
    # f.write(song.to_dict()['lyrics'])
    # f.close()
    return song.to_dict()['lyrics']

def find_avg_sentiment():
    # Create an instance of the sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Define the directory containing the text files
    directory = 'lyrics'

    # Initialize a dictionary to store average sentiment for each file
    average_sentiments = {}

    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # Consider only .txt files
            file_path = os.path.join(directory, filename)

            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Initialize variables to store cumulative sentiment scores
            total_sentiment = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
            num_lines = 0

            # Perform sentiment analysis for each line
            for line in lines[1:]:
                # Remove leading/trailing whitespaces and newline characters
                line = line.strip()

                # Perform sentiment analysis
                sentiment = sia.polarity_scores(line)

                # Add the sentiment scores to the cumulative totals
                total_sentiment['neg'] += sentiment['neg']
                total_sentiment['neu'] += sentiment['neu']
                total_sentiment['pos'] += sentiment['pos']
                total_sentiment['compound'] += sentiment['compound']

                # Increment the line counter
                num_lines += 1

            # Calculate the average sentiment scores
            avg_sentiment = {key: value / num_lines for key, value in total_sentiment.items()}

            # Store the average sentiment in the dictionary
            average_sentiments[filename] = avg_sentiment
    # return (average_sentiments)
    # Print the average sentiment for each file
    retu = []
    for filename, sentiment in average_sentiments.items():
        retu.append(list(sentiment.values()))
    return np.array(retu)
    # for filename, sentiment in average_sentiments.items():
    #     print("File:", filename)
    #     print("Average Sentiment:", sentiment)
    #     print()
 

token = get_token()
# result = search_for_artist(token, "Olivia Rodrigo")
# artist_id = result["id"]
# songs = get_songs_by_artist(token, artist_id)
# for idx, song in enumerate(songs):
#     print(f"{idx + 1}. {song['name'], song['id']} ")

# top = list(get_top_50(token))
# feat = get_feats(token, "7K3BhSpAxZBznislvUMVtn")
n = make_data(token)
print(n)
# print(top)
# np.savetxt("data.csv", n, delimiter=",")
# np.savetxt("name.csv", n, delimiter=",", fmt="%s")

# d = find_avg_sentiment()
# np.savetxt("senti.csv", d, delimiter=",")
# print(d)

# for filename, sentiment in d.items():
#     print("File:", filename)
#     print("Average Sentiment:", list(sentiment.values()))
#     print()


# l = lyrics(top)
# print(l)

