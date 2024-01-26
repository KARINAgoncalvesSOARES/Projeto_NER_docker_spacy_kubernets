#!/usr/bin/env python3
"""
Data Scientist Jr.: Karina Gonçalves Soares

NER and Kubernetes
==================
Neste script usaremos spaCy para o NER. Testaremos a nossa Aplicação
com Docker, logo publicaremos a Imagem gerada no Docker Hub e seguidamente 
a essa imagem será usada para o DEPLOYMENT em Kubernetes.

Nesse projeto utilizaremos a biblioteca de PLN Spacy do Python para realizar o Reconhecimento de Entidades Nomeadas(NER). 
Testaremos a nossa Aplicação com a plataforma Docker, logo publicaremos a Imagem gerada no Docker Hub e seguidamente 
essa imagem será usada para o DEPLOYMENT em Kubernetes onde a nossa aplicação que está no Spacy será automatizada.


Execução:

$ python app.py 
"""

# Classe Flask do módulo Flask, a função 'render_template' permite criar páginas web dinâmicas.
# A função 'request' é usada para acessar dados da requisição feita pelo cliente.
from flask import Flask, render_template, request
import spacy # Spacy é uma biblioteca que nos force ferramentas para trabalhar com Processamento de Linguagem Natural (PNL)

app = Flask(__name__) # Cria instância da classe Flask chamada 'app', o argumento '__name__' indica o nome do módulo atual.

# Carregar o modelo pré-treinado para o português:
# https://spacy.io/models/pt
nlp = spacy.load('pt_core_news_lg')

def exibir_tipos_entidades():
    '''Obter os tipos de entidades disponíveis no modelo'''
    tipos_entidades = nlp.get_pipe('ner').labels # Obtém os rótulos(labels) das entidades reconhecidas pelo modelo
    return tipos_entidades # A função retorna a lista de entidades disponíveis no modelo

# Define uma função que aceita um parâmetro chamado texto, isso indica que a função espera receber um texto como entrada.
def reconhecer_entidades(texto):
    '''Processar o texto usando o modelo do spaCy'''
    doc = nlp(texto) # Representação estruturada do texto

    # Exibir as entidades reconhecidas:
    # O trecho a seguir faz parte da função 'reconhecer_entidades' e está relacionada a extração de entidades nomeadas do texto processado pelo Spacy.
    entidades = [(entidade.text, entidade.label_) for entidade in doc.ents]
    return entidades

#O objetivo geral é obter o produto final (renderizar) da página inicial do aplicativo web
# Parte do aplicativo web usando Flask
@app.route('/') # Um decorador do Flask que chama a função 'index()'
def index():
    return render_template('index.html', tipos_entidades=exibir_tipos_entidades()) # Retorna o resultado da função 'render_template'

# Lida com a submissão de um formulário contendo texto, processa o texto para identificar entidades com a função 'reconhecer_entidades()' e exibe o resultado.
@app.route('/resultado', methods=['POST'])
def resultado():
    texto_usuario = request.form['texto']
    entidades = reconhecer_entidades(texto_usuario)
    return render_template('resultado.html', texto=texto_usuario, entidades=entidades, tipos_entidades=exibir_tipos_entidades())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)