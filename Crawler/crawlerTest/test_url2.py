#  -*- coding: utf-8 -*-

import urllib.request

url = "http://www.baidu.com"

print("first method")
response1 = urllib.request.urlopen(url)
print(response1.getcode())
print(len(response1.read()))

print("second method")
request = urllib.request.Request(url)
request.add_header("user-agent","Mozilla/5.0")
response1 = urllib.request.urlopen(url)
print(response1.getcode())
print(len(response1.read()))

