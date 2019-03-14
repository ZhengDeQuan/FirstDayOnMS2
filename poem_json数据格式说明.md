#~/Data/Poem/processed_poem.json

poems = [A,….A]

A  = poem_dict

poem_dict ={
"poem_id": int_type
"poem_title":str_type 
"origin_poem":str_type
"url":str_type
"washed_poem":str_type
"paras":list_type = B
"festival":list[str,….,str] 标志一个散文中是否有教师节，元宵节这种节日信息
"season":list[str,….,str] 标志一个散文中是否有春天，秋天，这种季节信息
}

B = [para_dict,……,para_dict] 有的情况下，一个文章，可以分成几个小文章，比如篇一：篇二：这种。如果这个文章不可以分成更小的文章的话，B这个list的长度为1。注意将大文章分成几个小文章的逻辑跟大文章中的自然段没有关系。以前数据处理中，将两者混为一谈的做法，已经被改正了。

para_dict ={ 
"para_title":str_type
"para_content":[
			自然段1,自然段2,…….,自然段n 
			]
每个小文章，也可能会有自然段。这个自然段的信息被保留了下来，这样如果需要对文章进行summarize的话，可以根据一个自然段作为最小的单元，而不是一个句子。这样可以更好的保证语义连贯性。

"fencied_para_title":list[str]
"fencied_para_content":[
					[str,….,str]自然段1的分词结果
					[str,….,str]
					.
					.
					.
					[str,….,str]自然段n的分词结果
				     ]
"key_words":[(str1,float1),(str2,float2),……,(str10,float10)] float是对应的关键词的权重，其值在(0,1]范围内。每个key_words列表对应的是目前这个小文章的关键词。目前只取前十个。抽取关键词的算法是TextRank。目前只允许名词，名动词(比如“思索”这种词)，地名（比如“故宫”这种词）这三种属性的词被抽出。
"festival":list[str,….str] 
"season":list[str,…..str] 这里的标记是针对这个小散文的标记，外边的大散文的标记，是内部各个小散文的标        记的并集
}
