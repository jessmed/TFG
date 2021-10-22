# -*- coding: utf-8 -*-
"""
Crea el archivo html con los tópicos para la aplicación web

@author: medye
"""

import json
import os
#Gensim
import gensim

# Load a potentially pretrained model from disk.
from gensim.test.utils import datapath

# Rutas de los archivos necesarios
WORK_DIR = os.path.dirname(os.path.abspath(__file__)) 
w = WORK_DIR + "\\DocViewer\\mi_aplicacion\\templates\\topics.html"
q = WORK_DIR + "\\final_database\\topic_color.json"


NUM_TOPICS = 15

# Cargamos el modelo del LDA
temp_file = datapath("model")
lda = gensim.models.LdaModel.load(temp_file)
print("Modelo cargado")


# Obtenemos matriz de los tópicos del modelo

matrix = lda.show_topics(num_topics=NUM_TOPICS, num_words=10, log=False, formatted=False)

# print(matrix)

color  = json.load(open(q))
html = "<div>\n"
html += "\t<p style=\"background-color:"+color[str(0)]+"; font-weight:bold;\">TOPIC "+str(0)+":</p>\n"
html +="\t<ul>\n"
html += "\t\t<li> Unknow topic</li>\n"
html+="\t</ul>\n"

for i in range(NUM_TOPICS):
    html += "\t<p style=\"background-color:"+color[str(i+1)]+"; font-weight:bold;\">TOPIC "+str(i+1)+":</p>\n"
    html +="\t<ul>\n"
    
    term_list = matrix[i]
    for term in term_list[1]:
        html += "\t\t<li>"+term[0]+"</li>\n"
    html+="\t</ul>\n"

html += "</div>"

f = open(w,'w')
f.write(html)
f.close()