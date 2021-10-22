# mi_aplicacion/views.py

from django.shortcuts import render, HttpResponse,redirect
from django.views import generic
import os
import json
WORK_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))


# Vista para el índice del viewer
def index(request):
    dir = WORK_DIR
    return render(request,'index.html')

# Vista para la lista de documentos
def doclist(request):

    path = WORK_DIR + "\\final_database\\text"

    # Eliminamos la extensió para mostrarlo como lista
    docs = [os.path.splitext(filename)[0] for filename in os.listdir(path)]
    # Dividimos la lista en 3 partes para mostrarla facilmente ya que son muchos archivos
    n = len(docs)//3
    docs_list = [docs[i:i + n] for i in range(0, len(docs), n)]

    return render(request, 'doclist.html', {'docs_1': docs_list[0],
                                            'docs_2': docs_list[1],
                                            'docs_3': docs_list[2]})


# Vista para mostrar las estadisticas que se obtienen del algoritmo LDA
def statistics(request):
    s = "stat.html"

    return render(request,'statistics.html',{'s': s})


# Vista para mostrar los documentos, leemos el JSON asociado al que se clica
# extraemos los campos del JSON para devolverlos
def document(request):
    # Obtenemos el nombre del paper que se ha usado
    paper_id = request.GET.get("doc")
    mode = request.GET.get("mode")
    path = WORK_DIR + "\\final_database\\"+mode+"\\"+paper_id+".json"

    path_colors = WORK_DIR + "\\final_database\\topic_color.json"

    json_data = open(path)

    json_colors = open(path_colors)
    obj = json.load(json_data)  # deserialises it
    obj_colors = json.load(json_colors)


    title = obj["title"]
    abstract = obj["abstract"]


    json_data.close()

    return render(request, 'document_base.html', {'paper_id': paper_id,
                                                  'title':title,
                                                  'abstract':abstract,
                                                  'mode':mode,
                                                  'obj_colors':obj_colors})


#-------------------------------------------------------------------------