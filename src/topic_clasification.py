# -*- coding: utf-8 -*-
"""

CLASIFICACIÓN DE DOCUMENTOS A PARTIR DEL MODELO GENERADO POR LDA

Cargamos el modelo generado y a partir de las funciones del modelo obtenemos
los datos necesarios para calcular la temática más probable para los dos modos
de los documentos: TEXT y PHRASES


@author: Jesús Medina Taboada
"""  

import time
import numpy as np  #version 1.20.3
import json
import os

#Gensim

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

#spacy
import spacy
from nltk.corpus import stopwords
#vis
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from os import listdir
# Load a potentially pretrained model from disk.
from gensim.test.utils import datapath

# --------------------------------------------------------------

WORK_DIR = os.path.dirname(os.path.abspath(__file__))   # Directorio actual de trabajo                                          # Aplicar filtro de palabras comunes    
                                           
NUM_TOPICS=15       # Número de documentos que usamos
# Cargamos json tipic-colores
color_list  = json.load(open(WORK_DIR + "\\final_database\\topic_color.json"))
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])



"""
    Para cada documento de la base de datos realizaremos los siguientes pasos:
        1. Cargamos el modelo preentrenado LDA
        2. Para cada documento de la base de datos lo convertimos en un BoW 
           para poder usar los métodos del modelo
        3. Calculamos la probabilidad para cada documento o frase de cada documento
        4. Lo añadimos a los archivos para que puedan leerse correctamente por el servidor
"""

stopwords = stopwords.words("english")
stopwords.extend(['well','such','other','as','many','have','from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


# # =============================================================================
# # LEMMATIZACION
# # =============================================================================

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    texts_out = []
    doc = nlp(texts)
    new_text = []
    for token in doc:
        if token.pos_ in allowed_postags:
            new_text.append(token.lemma_) 
            
    final = " ".join(new_text)
    texts_out.append(final)
    
    return (texts_out)

def gen_words(lenm_texts):
    word_list = [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in lenm_texts] 
   
    return (word_list)

def term2id(dic,term):
    for i,v in dic.token2id.items():
        if i == term:
            return v
        
    

# =============================================================================
#   1. CARGAMOS MODELO PREENTRENADO LDA
# =============================================================================
temp_file = datapath("model")
lda = gensim.models.LdaModel.load(temp_file)
print("Modelo cargado")
d = WORK_DIR + "\\final_database\\lda_dictionary\\dictionary.txtdic"
load_dic = corpora.Dictionary.load(d)


# =============================================================================
#   2. PARA CADA DOCUMENTO DE LA BASE DE DATOS
# =============================================================================

# Función que convierte un texto en una salida para calcular probabilidad
# de pertenecer a un tópico

def doc_2_bow(textos):
    lemmatized_texts = lemmatization(textos)
    data_words = gen_words(lemmatized_texts)
    dic = corpora.Dictionary(data_words)

    corpus = []
    for text in data_words:
        new = dic.doc2bow(text)
        corpus.append(new)
    return corpus



# Medimos tiempo inicio ejecución
inicio = time.time()

# =============================================================================
# CALCULO DE LA PROBABILIDAD
# =============================================================================

#                       TEXTOS COMPLETOS
w = WORK_DIR + "\\final_database\\text"
onlyfiles = listdir(w)
cont = 1
CICLOS = len(onlyfiles) 
# Para cada uno de los archivos de la base de datos TEXT
for file in onlyfiles:

    if cont % 10000 == 0:
        print("{}/{}".format(cont,CICLOS))
    
    obj  = json.load(open(w+"/"+file))
    # Para cada sección text del abstract(si es modo text solo tiene 1 sección)
    for t in obj["abstract"]:
        text = t["text"]
        # Lo convertimos en BoW
        corpus = doc_2_bow(text)
        # Obtenemos del modelo la matriz de probabilidad topico por documento
        topic_distribution_document = lda.get_document_topics(corpus,minimum_probability=None)
        # Obtenemos el topic on mayor probabilidad 
        main_topic = str(max(topic_distribution_document[0],key=lambda item:item[1])[0])
        # Añadimos tópic y color asociado al JSON del documento
        obj["abstract"][0]["topic"]= main_topic 
        obj["abstract"][0]["color"] = color_list[str(main_topic)] 
        # Guardamos doc JSON preprocesado en nueva carpeta  TEXTO COMPLETO                                  
        open(w+"/"+file, "w").write(
            json.dumps(obj, indent=4, separators=(',', ': '))
        )
    cont +=1
    
    
print("TEXTOS COMPLETOS CLASIFICADOS")    



# TEXTOS POR FRASES
q = WORK_DIR + "\\final_database\\phrases"
onlyfiles = listdir(q)
cont = 1
# Para cada uno de los archivos de la base de datos TEXT

for file in onlyfiles: 
    if cont % 100 == 0:
        print(cont)
    
    obj  = json.load(open(q+"/"+file))
  
    # Para cada sección text del abstract(si es modo text solo tiene 1 sección)
    for t in obj["abstract"]:
        
        text = t["text"]
        # Obtenemos lista de términos
        term_list = gen_words(lemmatization(text))
        # Obtenemos corpus del documento
        corpus = doc_2_bow(text)
        # Obtenemos del modelo la lista de probabilidad topico por documento
        p_x_d = lda.get_document_topics(corpus,minimum_probability=None)
        # Obtenemos del modelo la matriz de probabilidad término por tópico
        p_t_x = lda.get_topics()
        
        
        # Creamos la matriz probabilidad(numero topicos del modelo/número terminos de la frase)
        matrix_p = np.zeros((NUM_TOPICS,len(term_list[0])))
        
        # Para cada palabra lemmatizada de la frase miramos sus probabilidades de pertenecer a un tópico
        # usaremos un filtro de 1e-8
        
        col = 0
        for w in term_list[0]:
            # Calculamos el identificador del término en el corpus
            w_id = term2id(load_dic, w)
            # Si el término existe en el corpus calculamos probabilidad
            if w_id != None:
                # Para cada tópico que aparezca en relación al documento(que aparezcan)
                for k in p_x_d[0]:
                    
                    n_topic, prob = k
                    
                    # Para un término calculamos su probabilidad segun el tópico                       
                    prob_termino= prob * p_t_x[n_topic][w_id]
                    matrix_p[n_topic][col] = prob_termino
            col+=1
            
        # Normalizamos
        suma = 0
        # Sumamos la columna de la matriz(término/tópico)
        # si es 1 no lo sumamos y luego dividimos la suma entre
        s = np.sum(matrix_p, axis=0)
        # Normalizamos dividiendo cada columnda por el total
        filas,columnas = matrix_p.shape
        
        for i in range(filas):
            for j in range(columnas):
                if matrix_p[i,j] != 0:
                    matrix_p[i][j] = matrix_p[i][j] / s[j]
        
        
         
        # Aplicamos logaritmos para evitar underflow 
        matrix_p=np.log(matrix_p,where=0<matrix_p)
        matrix_p = np.sum(matrix_p, axis=1)
        
        # Se puede dar el caso que la frase este entera conformada por palabras fuera del
        # corpus del modelo, en cuyo caso no pertenecen a ningun topic y no se las puede clasifica
        # le damos entonces topic 0 que es indefinido
        total = np.sum(matrix_p)
                 

        if total != 0:
            main_topic = matrix_p.argmax() +1
            
        else:
            main_topic=0
        
        
        
        # Añadimos tópic y color asociado al JSON del documento
        t["topic"]= str(main_topic) 
        t["color"] = color_list[str(main_topic)]
    
    # Guardamos doc JSON preprocesado en nueva carpeta  TEXTO COMPLETO                                  
    open(q+"/"+file, "w").write(
        json.dumps(obj, indent=4, separators=(',', ': '))
    )
    
    cont +=1
    
print("TEXTOS POR FRASES CLASIFICADOS")     



