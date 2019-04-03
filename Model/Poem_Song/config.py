opt = {
    'song_threshold':0.50,#弃用
    'poem_threshold':0.001,#诗歌与关键词的匹配程度的最低阈值
    'poem_song_match_threshold':0.001,#弃用
    'lower_bound':3, #散文可以切割成的段数，如果只设置lower_bound,那么就只要切割成3段的，如果设置了lower_bound和upper_bound,那么段数在两个阈值的闭区间中的会被选中
    'upper_bound':None#如果设置,必须为int，且要大于lower_bound
}