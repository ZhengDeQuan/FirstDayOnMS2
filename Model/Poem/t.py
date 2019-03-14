import pickle
import re
import os

l = os.listdir("sanwenji")
print(l)

name  = ['aiqing', 'chunjie', 'dongtian', 'gudai', 'guoqingjie', 'guxiang', 'jiaoshijie', 'jieri', 'jingmei',
        'jishi', 'meng', 'mingjia', 'muai', 'qingchun', 'qinggangushi', 'qinggan', 'qingmingjie', 'qingrenjie',
         'qinqing', 'qiutian', 'qixi', 'shi', 'shanggan', 'shengdanjie', 'shuqing', 'xiatian', 'xiejing', 'xieren',
         'xiewu', 'xue', 'youmei', 'youqing', 'yuanxiaojie', 'zheli']

cname = ['爱情','春节','冬天','古代','国庆节','故乡','教师节','节日','精美',
        '记事','梦','名家','母爱','青春','情感故事','情感','清明节','情人节',
         '亲情','秋天','七夕','诗','伤感','圣诞节','抒情','夏天','写景','写人',
         '写物','雪','优美','友情','元宵节','哲理']

name = ['lizhi','shanggan', 'shuqing', 'suibi', 'xiandai', 'xiejing', 'youmeisanwen']
cname= ['励志','伤感','抒情','随笔','现代','写景','优美']
print(name)