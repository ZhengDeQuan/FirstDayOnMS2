from Model.Song.CheckCopyRight import music_search
import pickle
from tqdm import  tqdm
# sources_to_check = [music_search.MusicSource.MIGU, music_search.MusicSource.NETEASE]
sources_to_check = [music_search.MusicSource.MIGU]

if __name__ == "__main__":
    Checker = music_search.MusicSearch()
    all_songs = pickle.load(open("../song/all_songs_with_keywords.pkl","rb"))
    new_songs = []
    for song in  tqdm(all_songs):
        song_name = song['music_name']['format_name']
        singers = []
        for sig in song['singer_info']:
            singers.append(sig['format_name'])
        has_copyright_sources = Checker.check_copyright(name=song_name,singers=singers,sources_to_check=sources_to_check)
        if len(has_copyright_sources) > 0: #有版权
            new_songs.append(song)
    print('{} songs got copyright'.format(len(new_songs)))
    pickle.dump(new_songs,open("../song/all_songs_with_keywords_copyright.pkl","wb"))
