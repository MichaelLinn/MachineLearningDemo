#  -*- coding: utf-8 -*-


from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "http://www.pythonscraping.com/pages/page1.html"
# url = "http://www.baidu.com/"

html = urlopen(url)
bsObj = BeautifulSoup(html.read(), "lxml")
print(bsObj.head)




