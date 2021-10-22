# -*- coding: utf-8 -*-
"""

PREPROCESAMIENTO DE LOS PAPERS QUE CONFORMAN LA BASE DE DATOS
Los archivos tienen formato JSON, de los muchos campos de información que 
traen solo nos interesan 3 que son los que extaeremos.

    1.-'paper_id': número único de indetificación del documento
    2.- 'title': título del artículo(puede estar vacío)
    3.-'abstract': resumen del contenido del artículo y sobre el cual aplicaremos
                  los algoritmos del estudio a lrealizar
                  
Una vez filtrados, realizaremos una copia de cada archivo con una división
por frases para poder determinar en la fase final del proyecto el tema por frases
apartes de por el conjunto total de texto.


@author: Jesús Medina Taboada
"""                                                           


              
import json
import time
import os

# Biblioteca para hacer la división de texto en frases
import nltk
nltk.download('punkt')
from tokenize import tokenize

# Medimos tiempo inicio ejecución
inicio = time.time()

# Directorio actual de trabajo
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta del la base de datos de papers
path_docs = WORK_DIR + "\pdf_json"

# Ruta de carpeta donde guardaremos los documentos prprocesados(si no existe se crea)
if not os.path.exists(WORK_DIR + "\\final_database\\text"):
    os.makedirs(WORK_DIR + "\\final_database\\text")
path_preprocess_text = WORK_DIR + "\\final_database\\text"

if not os.path.exists(WORK_DIR + "\\final_database\\phrases"):
    os.makedirs(WORK_DIR + "\\final_database\\phrases")
path_preprocess_phrases = WORK_DIR + "\\final_database\\phrases"



# Listamos todos los archivos del directorio con los documentos
onlyfiles = os.listdir(path_docs)

# Variables para marcadores durante la ejecución y tiempo que tarda
cont = 1
l = len(onlyfiles)
inicio = time.time()    # Medimos tiempo inicio ejecución
cont_2=1

# Para cada archivo de la lista lo cargamos como json, nos quedamos con los 
# campos que nos interesan y guardamos el resultado en la nueva carpeta
for file in onlyfiles:
    # if cont > 5000:
    #     break
    cont_2+=1
    obj  = json.load(open(path_docs+"\\"+file))     # Documento original
    

    
    # Si el abstract está vacío o comienza por citations o es
    preprocesed_document_text = {}                  # Documento preprocessado texto entero
    preprocesed_document_phrases = {}               # Documento preprocessado por frases
    

    # Metemos paper_id y title en ambos
    preprocesed_document_text["paper_id"] = obj["paper_id"]
    preprocesed_document_phrases["paper_id"] = obj["paper_id"]

    preprocesed_document_text["title"] = obj["metadata"]["title"]
    preprocesed_document_phrases["title"] = obj["metadata"]["title"]
    
    #Metemos bastrac en text y phrases de diferente forma
    
    # Como el abstract puede estar conformado por varios párrafos vamos a unarlos todos
    # en uno para poder aplicar los algoritmos
    whole_text = ""
    preprocesed_document_text["abstract"]= []
    preprocesed_document_phrases["abstract"]= []
    
    # Juntamos todos los parrafos del abstract
    for subkey in obj["abstract"]:
        whole_text += subkey["text"]
        
    # FILTRADO DE DOCUMENTOS
    # Algunos documentos no tienen abstract o no son representativos porque
    # son citaciones o están incompletos, por ello los eliminamos
    
    if len(whole_text) > 150 and whole_text[:8].upper() != "CITATION":
    
        # TEXTO
        # Añadimos texto completo a los documentos  TEX
        preprocesed_document_text["abstract"].append({"text":whole_text})
    
        # FRASES
        # A partir del abstract completo lo dividimos en frases usando la biblioteca nltk
        # Buscamos cuantas frases hay en el texto entero, usamos como separador los puntos
        phrase_list = tokenize.sent_tokenize(whole_text)
 
        
        for element in phrase_list:
            # Condición para evitar que se cuelen espacios despues de los puntos
            if len(element) > 0:
                preprocesed_document_phrases["abstract"].append({"text":element})
        
    
        # Guardamos doc JSON preprocesado en nueva carpeta  TEXTO COMPLETO                                  
        open(path_preprocess_text+"\\"+file, "w").write(
            json.dumps(preprocesed_document_text, indent=4, separators=(',', ': '))
        )
        # Guardamos doc JSON preprocesado en nueva carpeta  TEXTO POR FRASES                                  
        open(path_preprocess_phrases+"\\"+file, "w").write(
            json.dumps(preprocesed_document_phrases, indent=4, separators=(',', ': '))
        )
        
        cont+=1
        
    
    # Mensaje para seguimiento del preprocesamiento
    if(cont_2 % 500 == 0):
        print("Procesando file {}/{}".format(cont_2,l))
    

print("{}/{}".format(cont,cont_2))

# Medición fin de tiempo
fin = time.time()
total = fin - inicio
total_m = total / 60
total_h = total_m / 60

print(str(total) + " segundos") 
print(str(total_m) + " minutos")
print(str(total_h) + " horas")


