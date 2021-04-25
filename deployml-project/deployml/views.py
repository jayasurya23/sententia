from django.http.response import HttpResponse, StreamingHttpResponse
import json
from django.core.mail import send_mail
from django.conf.urls.static import static
import pickle
from django.shortcuts import render,redirect
from django.views import View 
from .mymodule import *

def index(req):
    keyword=req.GET.get('keyword')
    mail=req.GET.get('mail')
    print(keyword,mail)
    per=int(score(keyword))
    print(per)
    sentiment= "Positive" if per>50 else "Negative" 
    #
    return render(req,"index.html",{'score':int(per),'place':keyword,'senti':sentiment})

def result(req):

    # loaded_model = pickle.load(open("pickle_model.pkl", 'rb'))
    # tfidf = pickle.load(open("tf.pkl", 'rb'))
    # mail=req.GET('mail')
    return HttpResponse
