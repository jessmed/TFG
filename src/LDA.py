# -*- coding: utf-8 -*-
"""

GENERACIÓN DEL MODELO LDA

Transformamos los textos obtenidos en el preprocesamiento de forma que el algoritmo
LDA puede tomarlos de entrada. Entrenamos el modelo y lo guardamos.

@author: medye
"""                                                           
import time
import json

#Gensim

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

#spacy
import spacy
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

#vis
import pyLDAvis
import pyLDAvis.gensim



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from os import listdir


# =============================================================================
# VARIABLES DE LA EJECUCIÓN
# =============================================================================
WORK_DIR = os.path.dirname(os.path.abspath(__file__))   # Directorio actual de trabajo                                       
NUM_TOPICS=15                                        # Número de temáticas que el modelo obtiene


# =============================================================================
# PREPARING DATA
# =============================================================================

# Medimos tiempo inicio ejecución
inicio = time.time()

stopwords = stopwords.words("english")
stopwords.extend(['well','such','other','as','many','have','from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])



w = WORK_DIR + "\\final_database\\text"
q = WORK_DIR + "\\final_database\\phrases"
d = WORK_DIR + "\\final_database\\lda_dictionary\\dictionary.txtdic"
onlyfiles = listdir(w)
CICLOS = len(onlyfiles)

# Para cada uno de los archivos que representan el articulo cientifico en formato JSON
# obtenemos el texto y lo añadimos a textos
print("-----------------CARGANDO ARCHIVOS---------------------------------")
textos=[]
cont = 1
for file in onlyfiles:
    obj  = json.load(open(w+"/"+file))
    l = obj["abstract"]
    textos.append(l[0]["text"])
    
    # # Mensaje para seguimiento del preprocesamiento
    if(cont % 50000 == 0):
        print("Cargando archivos {}/{}".format(cont,CICLOS))
    cont += 1

        
# # =============================================================================
# # LEMMATIZACION
# # =============================================================================

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []

    cont = 1
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
                
        # # Mensaje para seguimiento del preprocesamiento
        if(cont % 10000 == 0):
            print("Lemmatizando {}/{}".format(cont,CICLOS))
        cont += 1
                
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)


print("-----------------LEMMATIZACIÓN ARCHIVOS------------------------------")
lemmatized_texts = lemmatization(textos)
print (lemmatized_texts[1])    
 
# Realizamos una segmentación del los términos y aprovechamos para eliminar stopwords
def gen_words(lenm_texts):
    word_list = [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in lenm_texts] 
   
    return (word_list)

print("-----------------CONVERSIÓN LISTA DE PALABRAS------------------------")
data_words = gen_words(lemmatized_texts)


# Creamos diccionario de palabras únicas del modelo
id2word = corpora.Dictionary(data_words)
print(type(id2word))
print("DICIONARIO CREADO")
print("-------->",id2word)

# Descartamos las palabras problematicas por el extremo superior e infierior
print("EXTREMOS DESCARTADOS")
# Eliminamos extremos inferiores y superiores de frecuencia
id2word.filter_extremes(no_below=20, no_above=0.1)
print("-------->",id2word)
id2word.save(d, separately=None, sep_limit=10485760, ignore=frozenset({}), pickle_protocol=4)


# Creamos el corpus, estructura de datos formada por identificador y frecuencia de un término
print("-----------------CREACIÓN DEL CORPUS------------------------")
corpus = []
cont = 1
for text in data_words:
    new = id2word.doc2bow(text)
    corpus.append(new)
    
    # # Mensaje para seguimiento del preprocesamiento
    if(cont % 10000 == 0):
        print("Corpus file {}/{}".format(cont,CICLOS))
    cont += 1



# # =============================================================================
# # DEFINICIÓN DEL MODELO LDA
# # =============================================================================

print("Datos preparados con éxito")

# Medición fin de tiempo
fin2 = time.time()
total2 = fin2 - inicio
print(str(total2) + " segundos") 
print(str(total2 /60) + " minutos")

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=NUM_TOPICS,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha="auto")

print("Modelo LDA cargado con éxito")

# Guardamos el modelo

from gensim.test.utils import datapath

# Save model to disk.
temp_file = datapath("model")
lda_model.save(temp_file)

print("Modelo LDA guardado con éxito")
# Medición fin de tiempo
fin = time.time()
total = fin - inicio

print(str(total) + " segundos") 
print(str(total /60) + " minutos")

# # =============================================================================
# # VISUALIZACIÓN DE DATOS  
# # =============================================================================
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)


# Medición fin de tiempo
fin = time.time()
total = fin - inicio

print(str(total) + " segundos") 
print(str(total /60) + " minutos")
print(str(total /3600) + " horas")


# Guarda estadísticas
q = WORK_DIR + "\\final_database\\statistics\\statistics_{}_{}.html".format(CICLOS,NUM_TOPICS)
open(q, "w")

pyLDAvis.save_html(vis, q )

# Guardamos el último stat en el servidor web
c = WORK_DIR + "\\DocViewer\\mi_aplicacion\\templates\\stat.html"
open(c, "w")
pyLDAvis.save_html(vis, c )

  
pyLDAvis.show(vis, ip='127.0.0.1', port=8888, n_retries=50, local=True, open_browser=True, http_server=None) 


        
        
        
        
        
        
        
        
        
        
        
        
        