# Clustering des hymnes nationaux — Analyse non supervisée

Partitionnement textuel de 190 hymnes nationaux (traductions anglaises) par **K-Means** et **CAH Ward**, avec visualisations cartographiques et nuages de mots.

## Structure du projet

```
anthems.csv               Source : Kaggle (190 pays, 5 colonnes)
anthems_clustering.ipynb  Notebook principal — pipeline complet
anthems_clustered.csv     Dataset enrichi avec labels de clusters
map_kmeans.html           Carte interactive — clusters K-Means
map_hac.html              Carte interactive — clusters CAH
```

## Pipeline

1. **Pré-traitement NLP** — nettoyage des artefacts d'encodage, lemmatisation (WordNet), suppression des stopwords enrichis (formes archaïques des hymnes traduits)
2. **Vectorisation** — TF-IDF (1 734 features, bigrammes, `sublinear_tf`, normalisation L2)
3. **K-Means** — sélection du k optimal par méthode du coude + silhouette → k=5
4. **CAH Ward** — dendrogramme, silhouette → k=4 optimal
5. **Comparaison** — ARI, AMI, tableau de contingence, scores silhouette sur espace TF-IDF commun

## Résultats principaux

### K-Means (k=5) — clusters thématiques équilibrés

| Cluster | Thème dominant | Pays |
|---------|---------------|------|
| 0 | Guerre et résistance | 30 |
| 1 | Fraternité et unité | 46 |
| 2 | Nature et territoire | 56 |
| 3 | Foi et spiritualité | 49 |
| 4 | Liberté et patrie | 9 |

### CAH Ward (k=4) — structure hiérarchique asymétrique

La CAH révèle une structure fondamentalement différente : un grand tronc commun de 184 pays et **3 paires d'hymnes-outliers** isolés très tôt dans la hiérarchie :

- **{Chine, Macao}** — vocabulaire très spécifique, champ sémantique quasi unique
- **{Congo RC, Congo RDC}** — longueur et lexique atypiques
- **{Chypre, Grèce}** — similarité lexicale très forte entre eux, très distincte du reste

### Comparaison des approches

| Méthode | k optimal | Silhouette (TF-IDF) | Usage recommandé |
|---------|-----------|--------------------|--------------------|
| K-Means | 5 | 0.004 | Lecture thématique équilibrée |
| CAH Ward | 4 | — | Détection d'outliers, exploration hiérarchique |

> Les scores silhouette modestes sont normaux pour des espaces TF-IDF de haute dimension (creux, ~1 700 dimensions).

## Dépendances

```
pandas numpy scikit-learn scipy matplotlib seaborn
nltk wordcloud folium yellowbrick
```

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn nltk wordcloud folium yellowbrick
python -m nltk.downloader stopwords wordnet punkt punkt_tab
```

## Données

Source : [National Anthems — Kaggle](https://www.kaggle.com/), 190 hymnes traduits en anglais avec codes ISO et continent d'appartenance. Une valeur manquante sur `Alpha-2`.
