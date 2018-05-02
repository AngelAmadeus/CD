#AmadeusTwitter
import string
import pandas as pd
import tweepy as tw
import matplotlib.pyplot as plt
#%%Claves de la Aplicación para Twitter
Consumer_Key = ''
Consumer_Secret = ''
Access_Token = ''
Access_Token_Secret = ''
#%%Crear un archivo de autrizcion
auth = tw.OAuthHandler(Consumer_Key, Consumer_Secret)
auth.set_access_token(Access_Token, Access_Token_Secret)
api = tw.API(auth) #Esto me da el login
#%%Descargar los tweets de la persona de interes
people = ['@lopezobrador_','@RicardoAnayaC','@JoseAMeadeK','@JaimeRdzNL','@Mzavalagc']
all_tweets = [] # Solo se descargarán  200 tweets de cada persona
for pp in people:
    print('Descargando tweets de %s'%pp)
    new_tweets = api.user_timeline(screen_name=pp,count=200)
    all_tweets.extend(new_tweets)
del pp
#%% Obtención de información para el análisis
#tpm es solo una variable temporal para crear los dataframes con la información
tmp = [[tweet.user.screen_name, #Nombre del autor
        tweet.in_reply_to_screen_name, #Si fue repica a otro tweet
        tweet.retweet_count, #Cantidad de retweets
        tweet.created_at, #fecha de creacion
        tweet.text] for tweet in all_tweets] #texto de tweet
#for tweet in all_tweets:
#    tmp.extend([tweet.user.screen_name,
#        tweet.in_reply_to_screen_name,
#        tweet.retweet_count,
#        tweet.created_at,
#        tweet.text])
df_alltweets = pd.DataFrame(tmp) #Crear un dataframe con la informacion
df_alltweets.columns = ['posted_by', 'in_rp_name','retweet_count', 'created', 'text']
del tmp #Se elimina la variable temporal
#%% Funcion para quitar todos los signos de puntuacion
def remove_punctuation(x):
    x = str(x)
    try:
        x = ''.join(ch for ch in x if ch not in string.punctuation)
    except:
        pass
    return x
#%%Analisis del texto
cont = 0
while cont <= len(people)-1:
    pp = people[cont]#0-LObrador, 1-Anaya, 2-Meade, 3-Bronco, 4-Margarita
    tmp = list(df_alltweets.loc[df_alltweets.posted_by==pp[1:]].text)
    s_todo = ''
    for tweet in tmp:s_todo = s_todo + '' + tweet
    df_words = pd.DataFrame(s_todo.split(),columns=['words'])
    df_words.words = df_words.words.apply(remove_punctuation)
    s_limpia = []
    for word in df_words.words:
        if (len(word)>4)and(word.find('https')<0):
            s_limpia.append(word)
    df_words_clean = pd.DataFrame(s_limpia,columns=['words'])
    df_words_count = df_words_clean.words.value_counts()
    del s_limpia,s_todo,tweet,tmp,df_words,word
    cont+=1
    #Grafica
    plt.figure(figsize=(6,6))
    df_words_count.iloc[0:20].plot(kind='bar') #El top 20 de las palabras más comunes
    plt.ylabel('Frecuencia')
    plt.title('Conteo de palabras de %s'%pp)
    plt.show()