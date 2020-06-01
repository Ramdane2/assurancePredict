# Prédiction de la somme qu'une assurance va réserver pour un client en fonction de certains critères

# Objectif
l'objectif de ce travail est de déduire quelque somme une assurance vas facturer pour chaque clients.
le but est d’entraîner un algorithme d'apprentissage pour qu'il puissent déduire la somme d'assurance en fonction de fumeur ou non, de l'age et de l'indice de masse corporelle.

### Données

Le fichier data à été récupérer sur ce [lien](https://www.kaggle.com/mirichoi0218/insurance).
Ce fichier comporte 7 colonnes et 1338 lignes *(au passage 1338 lignes c'est très léger ce qui vas impacter la précision par la suite)*

### Préparation de la données
Je rappel que nous cherchons le coût que l'assurance va facturée à ces client
d’après notre fichier de données cette info ce situe dans la colonne **charges**
La question crucial, c'est de savoir sont les quelles colonnes qui ont le plus de corrélation avec notre colonne **charges**.
Ma méthode à était de faire une matrice de corrélation, mais pour cela j'ai transformé les colonne de type string en integer. 

> **assurance** correspond au dataframe

    assurance.loc[(assurance.sex == "female"), "sex"] = 0
    assurance.loc[(assurance.sex == "male"), "sex"] = 1
    assurance.loc[(assurance.smoker == "yes"), "smoker"] = 1
    assurance.loc[(assurance.smoker == "no"), "smoker"] = 0

    assurance.loc[(assurance.region == "northeast"), "region"] = 1
    assurance.loc[(assurance.region == "southeast"), "region"] = 2
    assurance.loc[(assurance.region == "southwest"), "region"] = 3
    assurance.loc[(assurance.region == "northwest"), "region"] = 4

    assurance[["sex", "smoker", "region"]] = assurance[["sex", "smoker", "region"]].apply(pd.to_numeric)

### Matrice de corrélation

    matrice_corr = assurance.corr()
    sns.heatmap(data=matrice_corr, annot=True)
   Nous pouvons constater que les colonnes smoker , age et bmi (Body Mass Index) on le plus grand indice de corrélation avec notre objectif qui de déduire la **charges**
   Je vais partir sur la prédiction en fonction de ces trois critères

### Décomposition en X (paramètres) et Y (objectif)

    X = pd.DataFrame(np.c_[assurance['smoker'],assurance['age'],assurance['bmi']], columns = ['smoker','age','bmi'])
    Y = assurance['charges']

### Base d'apprentissage par régression linéaire
*Décomposition 80% pour l'apprentissage et 20% pour les tests*

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
regression = LinearRegression()
regression.fit(X_train, Y_train)


### Score de l'apprentissage et des tests
*Model evaluation for training*
  

      y_train_predict = regression.predict(X_train)
        r2 = r2_score(Y_train, y_train_predict)
         print('score trainning  {}'.format(r2))

 
*model evaluation for testing set*

    y_test_predict = regression.predict(X_test)
    r2 = r2_score(Y_test, y_test_predict)
    print('score test {}'.format(r2))

Pour le training nous avons un score de 73,86% et 77,92% pour les tests, ces scores sont assez faible du au manque de quantité de données dans le fichier.

### Génération du model 
Cette ligne va permettre de générer le model utilisable directe sans passer par l'apprentissage.
   

     pickle.dump(regression, open('model.pkl','wb'))




## Installation 

Pour l'installation rien de plus simple, il suffit de récupérer le 

    git clone projethttps://github.com/Ramdane2/assurancePredict.git
    cd assurancePredict

Lancer l'installation des dépendances :
	

    sudo apt install python3-pip
    sudo pip3 install -r requirements.txt


  

### Exécution du programme 

Pour lancer le programme, il suffit de l'éxecuté **app.py**

    python3 app.py

Maintenant le serveur est prés, pour y'accéder il faut saisir sur internet **http://addresseVM:5000**
Il faut bien vérifier que le **port 5000** est ouvert pour permettre l’accès depuis internet au serveur Flask

Pour m'as part j'ai déployé l'application sur la plateforme Heroku disponible de puis ce [lien](https://assuranceprediction.herokuapp.com/)





