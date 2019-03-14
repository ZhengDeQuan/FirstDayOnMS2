import json
from elasticsearch import Elasticsearch
import copy

from Song.config import CrawlerEsHost, EsUser, EsPassword


class CommentProvider:
    _es = Elasticsearch(hosts=[CrawlerEsHost], http_auth=(EsUser, EsPassword))
    _index = "crawler_163music_song"
    _doc_type = "music"
    _default_source_include = ["id", "name", "artist", "comments"]
    _body_schema = {
        "query": {
            "bool": {
                "must": [
                ]
            }
        }
    }

    @classmethod
    def get_comments(cls, ori_name, ori_singers, ori_album):
        body = cls._make_body(ori_name, ori_singers, ori_album)
        result = cls._es.search(index=cls._index, doc_type=cls._doc_type, body=body,
                                _source_include=cls._default_source_include)
        comments = cls._process_result(result)
        return comments

    @classmethod
    def _make_body(cls, ori_name, ori_singers, ori_album):
        body = copy.deepcopy(cls._body_schema)
        if ori_name:
            body["query"]["bool"]["must"].append({"term": {"name.keyword": ori_name}})
        if ori_singers:
            body["query"]["bool"]["should"] = []
            for singer in ori_singers:
                body["query"]["bool"]["should"].append({"term": {"artist.name.keyword": singer}})
            body["query"]["bool"]["minimum_should_match"] = 1
        if ori_album:
            body["query"]["bool"]["must"].append({"term": {"album.name.keyword": ori_album}})
        return body


    @classmethod
    def _process_result(cls, result):
        music = None
        if result["hits"]["total"] == 1:
            music = result["hits"]["hits"][0]["_source"]
        elif result["hits"]["total"] > 1:
            music = cls._choose_music(result["hits"]["hits"])
        if not music:
            return [], []
        comments, replied_comments = cls._extract_comments_content(music["comments"])
        return comments, replied_comments
        #前者是所有的评论;后者是beReplied字段的内容
        #每个评论中的beReplied字段，记录的是本comment被那一条comment又评论了
        #我们需要在前者中剔除后者的内容

    @classmethod
    def _choose_music(cls, musics):
        assert len(musics) > 0
        selected_music = musics[0]
        for music in musics:
            if len(music["_source"]["comments"]) > len(selected_music["_source"]["comments"]):
                selected_music = music
        return selected_music["_source"]

    @classmethod
    def _extract_comments_content(cls, comments):
        if not comments:
            return [], []

        raw_comments = []
        replied_comments = []
        for comment in comments:
            raw_comments.append(comment["content"])
            replied = [it["content"] for it in comment["beReplied"]]
            replied_comments.extend(replied)
        return raw_comments, replied_comments

    @classmethod
    def batch_get_comments(cls, musics):
        for batch_musics in cls._get_batch_musics(musics):
            bodies = []
            for music in batch_musics:
                head = {"index": cls._index, "type": cls._doc_type}
                body = cls._make_body(*music)
                bodies.append(head)
                bodies.append(body)
            bodies = [json.dumps(it) for it in bodies]
            body_str = "\n".join(bodies) + "\n"
            result = cls._es.msearch(body=body_str)
            for response in result["responses"]:
                yield cls._process_result(response)



    @classmethod
    def _get_batch_musics(cls, musics, batch_size=20):
        batch_musics = []
        for idx, music in enumerate(musics):
            batch_musics.append(music)
            if (idx + 1) % batch_size == 0 or (idx + 1) == len(musics):
                yield batch_musics
                batch_musics = []

if __name__=="__main__":
    a = ['','','']#ori_name, ori_singers, ori_album
    b = [a,a,a,a,a]
    comments = CommentProvider.get_comments("aaaa","aaaaa","aaaaa")
    print(len(comments))
    for content in comments:
        print(content)
        print(len(content))
    # comments = list(CommentProvider.batch_get_comments(b))
    # print("*"*100)
    # print(len(comments))
    # for comment in comments:
    #     print(len(comment))
    #     for content in comment:
    #         print(content)
    #         print(len(content))

