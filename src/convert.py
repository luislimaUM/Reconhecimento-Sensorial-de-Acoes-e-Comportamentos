import extcolors
import webcolors
import os, sys
import webbrowser
import urllib.parse
from furl import furl

Map = {}

with open("colors_names.txt") as f:
    for line in f: 
        x = line.split('=',2)
        Map[x[0].strip()] = x[1].strip()

def rgbFromStr(s):  
        # s starts with a #.  
        r, g, b = int(s[1:3],16), int(s[3:5], 16),int(s[5:7], 16)  
        return r, g, b  

def closest_colour(color):
    R = color[0]
    G = color[1]
    B = color[2]
    mindiff = None
    for d in Map:  
        r, g, b = rgbFromStr(Map[d])  
        diff = abs(R -r)*256 + abs(G-g)* 256 + abs(B- b)* 256   
        if mindiff is None or diff < mindiff:  
            mindiff = diff  
            mincolorname = d  
    return mincolorname

def get_colour_name(requested_colour):
    try:
        return webcolors.rgb_to_name(requested_colour)
    except ValueError:
        return closest_colour(requested_colour)

def matrixToURL(m,file):
    base = 'https://www.extColors.html/' + file + '?'
    base = base + str(m[0][0]) + '=' + str(m[0][1]) 
    for line in range(1,len(m)):
            base = base + '&' + m[line][0] + '=' + str(m[line][1])
    return base 

def urlToMatrix(url):
    f = furl(url)
    dict = []
    colors = f.query.params
    for key,value in colors.items():
        dict.append([key,int(value)])
    #print(f.args['white'])
    return dict

def convertToMatrix(file):
        list = []
        colors, pixel_count = extcolors.extract(file)
        list.append(['total',pixel_count])
        for c in colors: 
                list.append([get_colour_name(c[0]),c[1]])
        return list

def openBrowser(url):
    webbrowser.open_new_tab(url)
    return

l = convertToMatrix(sys.argv[1])
#print(l)
url = matrixToURL(l,sys.argv[1])
#openBrowser(url)
dict = urlToMatrix(url)
#print(dict)