import extcolors
import webcolors
import os, sys
import math
from PIL import Image, ImageDraw
import time

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

#extract colors
colors, pixel_count = extcolors.extract(sys.argv[1])
#print results
extcolors.print_result(colors,pixel_count)
#create image
extcolors.image_result(colors, 150, sys.argv[1] + '_output')

for c in colors: 
    print(get_colour_name(c[0]))
