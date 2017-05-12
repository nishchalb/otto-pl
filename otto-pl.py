import spotipy
import spotipy.util as util
import numpy as np
import scipy.stats


def get_playlists(sp):
    """ Get the current user's playlists
    
    Args:
        sp (spotipy.client.Spotify): A spotipy client with an auth token

    Returns:
        list: A list containing spotify playlist objects represented as dicts
    """
    user = sp.current_user()
    playlists = []
    s_playlists = sp.user_playlists(user['id'])

    while s_playlists['next']:
        for playlist in s_playlists['items']:
            if ((not playlist['collaborative']) and 
                    playlist['owner']['id'] == user['id'] and
                    playlist['public'] and playlist['tracks']['total'] >1):
                playlists.append(playlist)
        s_playlists = sp.next(s_playlists)

    return playlists


def get_tracks_from_playlist(sp, playlist):
    """ Get a list of the tracks in a given playlist

    Args:
        sp (spotipy.client.Spotify): A spotipy client with an auth token
        playlist (dict): A dictionary representing a spotify playlist

    Returns:
        list: A list containing spotify track objects represented as dicts
    """
    user = sp.current_user()
    results = sp.user_playlist(user['id'], playlist['id'], fields='tracks,next')
    s_tracks = results['tracks']
    tracks = []

    while True:
        for track in s_tracks['items']:
            tracks.append(track['track'])
        if not s_tracks['next']:
            break
        s_tracks = sp.next(s_tracks)

    return filter(lambda t: t['id'] is not None, tracks)


def feature_vector_from_track(sp, track):
    """ Construct a feature vector from a given track

    Args:
        sp (spotipy.client.Spotify): A spotipy client with an auth token
        track (dict): A dictionary representing a spotify track object

    Returns:
        list: A list representing a feature vector for the given track
    """
    track_features = sp.audio_features([track['id']])[0]
    # Now construct vector
    vector = []
    vector.append(track_features['acousticness'])
    vector.append(track_features['danceability'])
    vector.append(track_features['energy'])
    vector.append(track_features['instrumentalness'])
    vector.append(track_features['key'])
    vector.append(track_features['liveness'])
    vector.append(track_features['loudness'])
    vector.append(track_features['mode'])
    vector.append(track_features['speechiness'])
    vector.append(track_features['tempo'])
    vector.append(track_features['time_signature'])
    vector.append(track_features['valence'])

    return vector


def use_user(username):
    token = util.prompt_for_user_token(username)
    if token:
        sp = spotipy.Spotify(auth=token)
        return sp
    else:
        print("cant get token for", username)


def get_distribution(sp, playlist):
    tracks = get_tracks_from_playlist(sp, playlist)
    num_features = len(feature_vector_from_track(sp, tracks[0]))
    data = np.zeros((len(tracks), num_features))
    for index, track in enumerate(tracks):
        vector = feature_vector_from_track(sp, track)
        data[index, :] = np.asarray(vector)
    mean = np.mean(data, axis=0)
    cov = np.cov(data.transpose())
    # return data
    return scipy.stats.multivariate_normal(mean=mean, cov=cov,
                                           allow_singular=True)


def log_likelihood_track_playlist(sp, track, playlist):
    print playlist['name']
    dist = get_distribution(sp, playlist)
    track_vector = feature_vector_from_track(sp, track)
    return dist.logpdf(track_vector)


def log_likelihoods_for_track(sp, track, playlists):
    results = []
    for playlist in playlists:
        likelihood = log_likelihood_track_playlist(sp, track, playlist)
        results.append((playlist['name'], likelihood))
    return sorted(results, key=lambda pair: pair[1])
