'''***************************************************************
    Chatbot con filtrado de respuestas y Verificación de respuesta
    **************************************************************'''

# 1. Importar bibliotecas y configurar clave
import openai
import os
import spacy
import numpy as np
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Creamos las variables a almacenar la historia
preguntas_anteriores=[]
respuestas_anteriores=[]
modelo_spacy = spacy.load("es_core_news_md")
# Lista de palabras prohibidas
palabras_prohibidas = ["madrid", "paris"]

# Calculo de similitudes
def similitud_coseno(vec1, vec2):
    superposicion = np.dot(vec1, vec2)
    magnitud1 = np.linalg.norm(vec1)
    magnitud2 = np.linalg.norm(vec2)
    sim_cos = superposicion/(magnitud1+magnitud2)
    return sim_cos

# Vectorizar las respuestas
def es_relevante(respuesta, entrada, umbral=0.5):
    entrada_vectorizada = modelo_spacy(entrada).vector
    respuesta_vectorizada = modelo_spacy(respuesta).vector
    similitud = similitud_coseno(entrada_vectorizada,respuesta_vectorizada)
    return similitud >= umbral


# Funcion para filtrar palabras prohibidas
def filtrar_lista_negra(texto, lista_negra):
    token = modelo_spacy(texto)
    resultado = []

    for t in token:
        if t.text.lower() not in lista_negra:
            resultado.append(t.text)
        else:
            resultado.append("[xxxx]")

    return " ".join(resultado)



# 2. Función para peticiones
def preguntar_chat_gpt(prompt, modelo="text-davinci-002"):
    respuesta = openai.Completion.create(
        engine=modelo,
        prompt=prompt,
        n=1,
        max_tokens=20,
        temperature=0.5
    )
    respuesta_sin_controlar = respuesta.choices[0].text.strip()
    respuesta_controlada = filtrar_lista_negra(respuesta_sin_controlar, palabras_prohibidas)
    return respuesta_controlada

# 3. Funcionamiento básico

print("Bienvenido al chat básico. Escribe 'salir' cuando queiras terminar")
while True:
    conversacion_historica=""

    ingreso_usuario =input("\nTú:")
    if ingreso_usuario.lower() == "salir":
        break

    # Construimos la conversacion historica
    for pregunta,respuesta in zip(preguntas_anteriores, respuestas_anteriores):
        conversacion_historica += f"El usuario pregunta: {pregunta}\n"
        conversacion_historica += f"ChatGPT responde: {respuesta}\n"

    prompt = f"El usuario pregunta: {ingreso_usuario}\n"
    conversacion_historica += prompt
    respuesta_gpt = preguntar_chat_gpt(conversacion_historica)

# Interceptar respuestas
    relevante = es_relevante(respuesta_gpt,ingreso_usuario)
    if relevante:
        print(f"{respuesta_gpt}")
        # Almacenar las conversaciones
        preguntas_anteriores.append(ingreso_usuario)
        respuestas_anteriores.append(respuesta_gpt)
    else:
        print("La respuesta no es relevante")

# 4. Ejecutar el Chatbot : Desde la terminal ejecutamos python Chatbot.py
