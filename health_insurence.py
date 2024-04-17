# Analisi di un dataset delle spese sanitarie in america

# Librerie
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# apertura file csv e stampa delle info generale del dataset
file_path = '/content/insurance2.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1',  sep=',')

# visualizzazione dati
data_info = data.info() # informazioni generali
data_head = data.head() # prime 5 righe
data_tail = data.tail() # ultime 5 righe

data_info, data_head, data_tail

# esplerazione del dataset e e visualizzazione dei dati
print(data.describe()) # Statistiche descrittive delle variabili numeriche
print(data.isnull().sum()) # Conteggio dei valori mancanti per ciascuna colonna

# Visualizzazione della distribuzione delle spese sanitarie (charges)
plt.figure(figsize=(10, 6))
sns.histplot(data['charges'], kde=True)
plt.title('Distribuzione delle Spese Sanitarie')
plt.xlabel('Spese')
plt.ylabel('Frequenza')
plt.show()

# Visualizzazione della relazione tra BMI (Body Mass Index) e spese sanitarie
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', data=data, hue='smoker')
plt.title('Relazione tra BMI e Spese Sanitarie differenziate per Fumatori')
plt.xlabel('BMI')
plt.ylabel('Spese')
plt.show()

# Visualizzazione della relazione tra age e spese sanitarie
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', data=data, hue='smoker')
plt.title('Relazione tra AGE e Spese Sanitarie differenziate per Fumatori')
plt.xlabel('AGE')
plt.ylabel('Spese')
plt.show()

# Boxplot per esplorare la distribuzione delle spese sanitarie tra diverse regioni
plt.figure(figsize=(10, 6))
sns.boxplot(x='region', y='charges', data=data)
plt.title('Distribuzione delle Spese Sanitarie per Regione')
plt.xlabel('Regione')
plt.ylabel('Spese')
plt.show()

# visualizzazione box plot dei dati numerici per la ricerca di outliers
plt.figure(figsize=(10, 6))
data.boxplot(column=['age', 'bmi', 'children', 'charges'])
plt.title('Box plot of Numerical Columns')
plt.show()

# Grafico a barre per visualizzare il numero di osservazioni per fumatori e non fumatori
plt.figure(figsize=(10, 6))
sns.countplot(x='smoker', data=data)
plt.title('Conteggio Fumatori e Non Fumatori')
plt.xlabel('Fumatore')
plt.ylabel('Conteggio')
plt.show()

# rimozione outliers
data = data[(data['charges'] < 21000) & (data['bmi'] < 46)] # scelta soglia per rimuovere gli outliers delle variabili numeriche

# boxplot dopo aver rimosso gli outliers
data.boxplot(column=['age', 'bmi', 'children', 'charges'])
plt.title('Box plot of Numerical Columns')
plt.show()

data_describe = data.describe()
data_describe

# visualizzazione dati senza outliers
# Visualizzazione della distribuzione delle spese sanitarie (charges)
plt.figure(figsize=(10, 6))
sns.histplot(data['charges'], kde=True)
plt.title('Distribuzione delle Spese Sanitarie')
plt.xlabel('Spese')
plt.ylabel('Frequenza')
plt.show()

# Distribuzione charges per fumatori e non
plt.figure(figsize=(10, 6))
data.boxplot(column='charges', by='smoker')
plt.xlabel('Smoker')
plt.ylabel('Charges')
plt.title('Distribuzione charges per fumatori e non')
plt.show()

# Visualizzazione della relazione tra BMI (Body Mass Index) e spese sanitarie
plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', data=data, hue='smoker')
plt.title('Relazione tra BMI e Spese Sanitarie differenziate per Fumatori')
plt.xlabel('BMI')
plt.ylabel('Spese')
plt.show()

# Visualizzazione della relazione tra age e spese sanitarie
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', data=data, hue='smoker')
plt.title('Relazione tra AGE e Spese Sanitarie differenziate per Fumatori')
plt.xlabel('AGE')
plt.ylabel('Spese')
plt.show()

# Boxplot per esplorare la distribuzione delle spese sanitarie tra diverse regioni
plt.figure(figsize=(10, 6))
sns.boxplot(x='region', y='charges', data=data)
plt.title('Distribuzione delle Spese Sanitarie per Regione')
plt.xlabel('Regione')
plt.ylabel('Spese')
plt.show()

# Grafico a barre per visualizzare il numero di osservazioni per fumatori e non fumatori
plt.figure(figsize=(10, 6))
sns.countplot(x='smoker', data=data)
plt.title('Conteggio Fumatori e Non Fumatori')
plt.xlabel('Fumatore')
plt.ylabel('Conteggio')
plt.show()

# Totale spese sanitarie per ogni regione
plt.figure(figsize=(10, 6))
charges_by_region = data.groupby('region')['charges'].sum()
plt.bar(charges_by_region.index, charges_by_region.values, color='red')
plt.xlabel('Regioni')
plt.ylabel('Totale spese')
plt.title('Totale spese per ogni Regione')
plt.xticks(rotation=45)
plt.show()

# contegio fumatori per regione
plt.figure(figsize=(10, 6))
smokers_by_region = data.groupby('region')['smoker'].value_counts().unstack().fillna(0)
smokers_by_region.plot(kind='barh')
plt.xlabel('numeri')
plt.ylabel('Regione')
plt.title('numeri di fumatori per regione')
plt.legend(title='Smoker', loc='upper right')
plt.show()

# plot features VS features
sns.pairplot(data, hue="smoker")

# Mappa di correlazione e stampa delle feature piu significative
data_numerica = pd.get_dummies(data, drop_first=True) # Convertire le variabili categoriche in variabili dummy (one-hot encoding)

# Calcolare nuovamente la matrice di correlazione utilizzando solo dati numerici
corr_numerica = data_numerica.corr()

# Creare la mappa di calore utilizzando la nuova matrice di correlazione
plt.figure(figsize=(12, 10))
sns.heatmap(corr_numerica, annot=True, cmap='coolwarm')
plt.title('Mappa di Calore della Correlazione')
plt.show()

# caratterististe rilevanti in base alla correlazione
threshold = 0.3
relevant_features = corr_numerica[(corr_numerica['charges'].abs() > threshold) & (corr_numerica.index != 'charges')].index.tolist()
print("Le featurs rilevanti in base alla correlazione sono:")
print(relevant_features)

#modello predittivo
X = data_numerica[['age', 'smoker_yes']] # input
y = data_numerica['charges']             # output

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizzazione
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# modello 1: Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)

# Tracciare i valori effettivi e quelli previsti per il modello 1
plt.figure(figsize=(8, 4))
plt.scatter(y_test, dt_predictions, color='green', label='Decision Tree')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Decision Tree: effettivi vs. predetti')
plt.xlabel('effettivi')
plt.ylabel('predetti')
plt.legend()
plt.show()

# stampa le metriche per valutare i modelli
print("Decision Tree - MSE: ", dt_mse)
print("Decision Tree - MAE: ", dt_mae)
print("Decision Tree - R2: ", dt_r2)

# importanza e peso di ogni feature in ogni modello analizzato

# Decision Tree
print("Decision Tree:")
importance = dt_model.feature_importances_

for i, feature in enumerate(X.columns):
    print(f"{feature}: {importance[i]}")
