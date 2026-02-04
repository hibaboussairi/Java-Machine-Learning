# 🤖 Mini Framework Machine Learning - Java

Un framework de Machine Learning orienté objet développé en Java, implémentant des algorithmes classiques de régression et classification.

## 📋 Description

Ce projet est un mini-framework de Machine Learning conçu dans le cadre d'un projet académique de Programmation Orientée Objet (POO). Il propose une architecture modulaire et extensible permettant d'entraîner et d'évaluer différents modèles de Machine Learning.

## ✨ Fonctionnalités

### Algorithmes Implémentés

- **Régression Linéaire** : Implémentation avec descente de gradient
- **KNN Régression** : K plus proches voisins pour la régression
- **KNN Classification** : K plus proches voisins pour la classification

### Outils de Prétraitement

- **MinMaxScaler** : Normalisation des données entre 0 et 1
- **StandardScaler** : Standardisation des données (moyenne = 0, écart-type = 1)

### Utilitaires

- **DataUtils** : Division train/test des données
- **Metrics** : Calcul de métriques (R², Accuracy)

## 🏗️ Architecture

```
code/
├── app/
│   └── Main.java              # Point d'entrée du programme
├── core/
│   └── MLModel.java           # Classe abstraite de base pour tous les modèles
├── linear/
│   └── LinearRegression.java  # Implémentation de la régression linéaire
├── knn/
│   ├── KNNRegression.java     # KNN pour la régression
│   └── KNNClassification.java # KNN pour la classification
├── preprocessing/
│   ├── Preprocessor.java      # Interface pour le prétraitement
│   ├── MinMaxScaler.java      # Normalisation Min-Max
│   └── StandardScaler.java    # Standardisation
├── metrics/
│   └── Metrics.java           # Métriques d'évaluation
└── model_selection/
    └── DataUtils.java         # Utilitaires pour la gestion des données
```

## 🚀 Installation et Utilisation

### Prérequis

- Java JDK 8 ou supérieur
- Un IDE Java (Eclipse, IntelliJ IDEA, VS Code, etc.)

### Compilation

```bash
# Naviguer vers le répertoire du projet
cd Java-ML/code

# Compiler tous les fichiers Java
javac -d bin app/*.java core/*.java linear/*.java knn/*.java preprocessing/*.java metrics/*.java model_selection/*.java

# Ou utiliser votre IDE pour compiler le projet
```

### Exécution

```bash
# Exécuter le programme principal
java -cp bin ml.app.Main
```

## 💡 Exemple d'Utilisation

```java
// Créer et entraîner un modèle de régression linéaire
LinearRegression lr = new LinearRegression(0.01, 1000);
lr.train(trainData);

// Faire des prédictions
double prediction = lr.predict(new double[]{5.0});

// Évaluer le modèle
double r2Score = lr.score(testData);
System.out.println("R² Score: " + r2Score);
```

## 📊 Résultats

Le programme principal (`Main.java`) effectue des tests comparatifs sur :

- Différents ratios de division train/test (20%, 30%)
- Différents taux d'apprentissage pour la régression linéaire (0.1, 0.01, 0.001)
- Différentes valeurs de k pour KNN (1, 3, 5, 7)
- Avec et sans prétraitement des données

## 🎯 Concepts POO Utilisés

- **Abstraction** : Classe abstraite `MLModel`
- **Héritage** : Tous les modèles héritent de `MLModel`
- **Polymorphisme** : Méthodes `train()`, `predict()`, `score()` redéfinies
- **Encapsulation** : Attributs privés avec getters/setters
- **Interfaces** : `Preprocessor` pour le prétraitement

## 📝 Métriques d'Évaluation

- **R² Score** : Pour les modèles de régression (mesure la qualité de l'ajustement)
- **Accuracy** : Pour les modèles de classification (taux de bonnes prédictions)


## 👥 Auteur
**Hiba Boussairi**

