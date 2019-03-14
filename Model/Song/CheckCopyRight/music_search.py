# coding: utf-8
from elasticsearch import Elasticsearch
# from config.account import ProdKgEsHost, EsUser, EsPassword
ProdKgEsHost, EsUser, EsPassword = 'http://corechat-usermemory-int.trafficmanager.net:19200/','esuser', 'Kibana123!'
import copy
from enum import Enum


class MusicSource(Enum):
    MIGU = "MIGU"
    NETEASE = "NETEASE"
    YEELIGHT = "YEELIGHT"

    def __str__(self):
        return self._name_
    
    @classmethod
    def has_name(cls, name):
        return any(name == item.name for item in cls)
    
    @classmethod
    def available_names(cls):
        return [item.name for item in cls]


class MusicSearch:
    def __init__(self):
        self._es = Elasticsearch(hosts=[ProdKgEsHost], http_auth=(EsUser, EsPassword))
        self._music_source_to_index = {
            MusicSource.NETEASE: "netease_music_merged_current",
            MusicSource.MIGU: "migu_music_merged_current",
            MusicSource.YEELIGHT: "yeelight_migu_music_current"
        }
        self._type = "e_music"
        self._body_schema = {
            "query": {
                "bool": {
                    "must": [
                    ]
                }
            }
        }
        self._source_include = ["music_name", "music_id", "can_play"]
        self._cache = dict()

    def _make_key(self, name, singers, sources_to_check):
        name_str = name if name else ""
        singer_str = "&".join(singers) if singers else ""
        sources = "&".join((str(it) for it in sources_to_check)) if sources_to_check else ""
        key = f"name:{name_str}+singer:{singer_str}+sources:{sources}"
        return key
        
    def check_copyright(self, name, singers, sources_to_check):#singer:list[format_singer(ele) for ele in singers] format_music(name) sources_to_check:list[MusicSource.MIGU,MusicSource.NETEASE]
        """Check the music sources with copyright.
        Return a list of `MusicSource`.
        """
        key = self._make_key(name, singers, sources_to_check)
        if key in self._cache:
            return self._cache[key]

        body = copy.deepcopy(self._body_schema)
        body["query"]["bool"]["must"].append({"match": {"can_play": True}})

        if name:
            body["query"]["bool"]["must"].append({"match": {"music_name.format_name": name}})
        if singers:
            for singer in singers:
                body["query"]["bool"]["must"].append({"match": {"singer_info.format_name": singer}})
        
        has_copyright_sources = []
        for source in sources_to_check:
            if source in self._music_source_to_index:
                index = self._music_source_to_index[source]
                result = self._es.search(index=index, doc_type=self._type, body=body, _source_include=self._source_include, size=1)
                if result["hits"]["total"] > 0:
                    has_copyright_sources.append(source)

        self._cache[key] = has_copyright_sources
        return has_copyright_sources