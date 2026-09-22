# Rapport technique — Churn Prediction

Document unique de référence du projet : audit initial, corrections apportées,
architecture, données, chaîne de qualité et guide de contribution.

Deux documents l'accompagnent : [README.md](README.md) (présentation et
résultats du projet) et [DEPLOYMENT.md](DEPLOYMENT.md) (mise en production).

---

## 1. Résumé

Le projet prédit le départ (churn) de clients bancaires à partir d'un dataset
de 10 000 clients. Un modèle LightGBM tuné atteint **PR-AUC 0.719** et
**AUC-ROC 0.871**, avec un recall de 89,9 % au seuil de décision calibré sur
le coût métier. Il est servi par une application Streamlit qui explique
chaque prédiction via SHAP.

Le projet est parti d'un prototype fonctionnel mais monolithique. Il est
aujourd'hui structuré en package testé, avec intégration continue et
déploiement automatique.

| Indicateur | Avant | Après |
|-----------|-------|-------|
| Violations PEP8 | 32 | 0 |
| Tests automatisés | 0 | 123 |
| Couverture | 0 % | 89 % |
| Modules réutilisables | 0 | 9 |
| Duplication du feature engineering | 3 endroits | 1, avec test de parité |
| CI/CD | aucune | lint + tests + sécurité + déploiement |
| Coût métier sur 2 000 clients | 224 808 € sans action | 70 574 € |

---

## 2. Audit initial

L'audit de départ notait le projet **6,2/10** : méthodologie ML solide, mais
dette structurelle importante.

### Points forts constatés

- Benchmark de 8 modèles avec comparaison équitable et split stratifié.
- Tuning par `RandomizedSearchCV`, sélection justifiée par le coût métier.
- Interprétabilité SHAP et seuil de décision calibré sur les coûts réels
  (faux négatif 500 €, faux positif 50 €).
- Feature engineering pertinent, justifié variable par variable.

### Faiblesses critiques

| Domaine | Problème | Gravité |
|---------|----------|---------|
| Qualité du code | 32 violations PEP8 dans un `app.py` de 230 lignes | Critique |
| Tests | Aucun test, aucune protection contre les régressions | Critique |
| Modularité | Tout dans un seul fichier ; feature engineering dupliqué entre l'app et les notebooks | Critique |
| Reproductibilité | Graines et chemins en dur, dépendances non figées | Haute |
| Gestion d'erreurs | `except Exception` génériques, aucun logging | Haute |
| Intégrité du modèle | Aucune validation au chargement | Moyenne |
| Déploiement | Manuel, sans conteneurisation ni CI | Moyenne |
| Monitoring | Aucune trace des prédictions servies | Moyenne |

---

## 3. Corrections apportées

### 3.1 Structure et qualité du code

Le monolithe `app.py` a été découpé en huit modules `src/`, et l'interface
réécrite dans `streamlit_app.py`. `app.py` a ensuite été supprimé : il
dupliquait intégralement la logique de l'application refactorisée.

`black` et `flake8` passent sans aucune violation, avec une configuration
partagée entre le poste de développement et la CI (`pyproject.toml`,
`.flake8`, `.pre-commit-config.yaml`).

### 3.2 Tests

123 tests couvrent 89 % du code :

| Fichier | Portée |
|---------|--------|
| `test_features.py` | Feature engineering, groupes d'âge et d'ancienneté, parité train/serve |
| `test_validation.py` | Schéma Pydantic, plages et énumérations |
| `test_model.py` | Chargement, validation d'intégrité, prédiction |
| `test_logger.py` | Configuration du logging, journal des prédictions |
| `test_exceptions.py` | Hiérarchie et chaînage des exceptions |
| `test_integration.py` | Pipeline complet sur le vrai modèle `best_model.pkl` |
| `test_decision.py` | Règle de rétention par valeur attendue |

### 3.3 Suppression du risque de train/serve skew

C'était le défaut le plus dangereux : le feature engineering existait en trois
exemplaires (notebook de traitement, `app.py`, `src/features.py`). Une
modification dans l'un sans report dans les autres aurait fait diverger
silencieusement les features d'entraînement de celles servies en production —
un modèle dégradé sans aucune erreur visible.

La correction tient en trois points :

1. `build_features_frame()` a été ajoutée à `src/features.py` : elle applique
   le feature engineering au dataset complet. Le notebook `02_processing`
   l'appelle désormais au lieu de réimplémenter les transformations.
2. Les seuils (`AGE_GROUPS`, `TENURE_GROUPS`, `PRODUCTS_HIGH_RISK_MIN`) ne
   vivent plus que dans `config.py`. Les deux chemins — un client, ou
   10 000 — les lisent.
3. `TestTrainServeParity` fait passer sept profils clients par les deux
   chemins et exige des vecteurs identiques. Toute divergence casse la CI.

**Vérification de non-régression** : au moment du refactor,
`build_features_frame()` régénérait `data/processed/churn_dataset_clean.csv`
**identique à l'octet près** au fichier ayant servi à entraîner le modèle en
production — colonnes, ordre et types compris. Le refactor lui-même n'a donc
rien changé au modèle. Le jeu de features a évolué ensuite, mais
délibérément (§ 3.10) et avec réentraînement.

### 3.4 Modalités catégorielles figées

`pd.get_dummies(drop_first=True)` déduit ses catégories de l'échantillon reçu.
Sur un jeu de données où l'Espagne serait absente, la colonne de référence
changeait — produisant des features incompatibles avec le modèle entraîné.
Les catégories de `country` et `gender` sont maintenant déclarées depuis
`config.py` avant l'encodage.

### 3.5 Notebooks

- **01_eda** : la cellule `kagglehub.dataset_download()` téléchargeait le
  dataset dans un cache jamais utilisé — le notebook lisait en réalité un
  fichier local, via un chemin Windows en dur (`..\data\raw\...`) incompatible
  avec Linux et macOS. Le chargement passe désormais par `RAW_DATA_PATH` et
  s'arrête avec un message explicite si le fichier manque. Une cellule qui
  réécrivait `data/raw/` à partir de son propre contenu a été supprimée.
- **02_processing** : appelle `build_features_frame()` au lieu de
  réimplémenter l'encodage et les features dérivées. Les justifications
  métier restent dans les cellules markdown.
- **03_modeling** : `random_state=42` (répété dix fois) remplacé par
  `RANDOM_STATE` importé de la config ; chemins de figures et de modèles
  dérivés de `config.py`. Les figures étaient écrites dans
  `notebooks/results/figures/` selon le répertoire d'exécution ; elles sont
  désormais regroupées dans `results/figures/`.

Les trois notebooks ont été ré-exécutés et leurs sorties correspondent au
code affiché. Le notebook 03 a d'abord été modifié sans réexécution — ses
changements étaient strictement équivalents (`RANDOM_STATE` vaut 42) — puis
réentraîné lors des évolutions décrites au § 3.10.

### 3.6 Robustesse : erreurs, logging, traçabilité

Six exceptions métier (`ModelNotFoundError`, `InvalidInputError`,
`SHAPExplainerError`…) remplacent les `Exception` génériques, et le logging
est centralisé dans `src/logger.py` avec rotation quotidienne.

Trois défauts d'intégration ont été corrigés dans l'application, qui déclarait
ces modules sans les utiliser :

- `streamlit_app.py` interceptait `FileNotFoundError` alors que `load_model()`
  lève `ModelNotFoundError`, qui n'en hérite pas. Le message d'erreur soigné
  était donc du code mort : un modèle manquant produisait une traceback brute
  à l'écran.
- L'application configurait son propre `logging.basicConfig()`, ce qui
  court-circuitait entièrement le logging centralisé — précisément là où il
  sert le plus.
- `PredictionLogger` n'était appelé nulle part : le journal des prédictions
  destiné à la détection de drift ne collectait rien.

Un quatrième défaut a été trouvé en exécutant le pipeline de bout en bout :
chaque module créait son propre `TimedRotatingFileHandler` sur le même
fichier. Sous Windows, la rotation de minuit échouait alors à chaque appel
(`PermissionError`, fichier verrouillé par un autre handler). Les handlers
sont désormais partagés — un seul descripteur ouvert par processus.

### 3.7 Reproductibilité et déploiement

- `RANDOM_STATE` et `TEST_SIZE` centralisés, versions figées dans
  `requirements.txt`, alignées sur l'environnement réellement testé.
- `Dockerfile` multi-stage sur `python:3.12-slim`, avec healthcheck.
- Deux workflows GitHub Actions (lint / tests / sécurité, puis vérification
  de déployabilité sur `main`).
- `.gitignore` complété : les artefacts générés (`.coverage`, `htmlcov/`,
  `logs/`, `.pytest_cache/`) ne risquent plus d'être committés.

### 3.8 Nomenclature et suppression des doublons

- **`requirements-lock.txt` supprimé.** Le fichier était encodé en UTF-16 et
  listait 337 paquets de l'environnement global de la machine (`anthropic`,
  `arabic-reshaper`…), pas ceux du projet : il donnait une fausse impression
  de reproductibilité. `requirements.txt` reste seul, avec toutes les versions
  figées et vérifiées contre l'environnement réellement testé. `pydantic` y a
  été reclassé en dépendance d'exécution — il était listé comme outil de
  développement alors que `src/validation.py` en dépend.
- **Bases MLflow orphelines supprimées.** Un `mlflow.db` traînait à la racine
  (0 run, base vide) et `notebooks/mlflow.db` (1 run) n'était référencé par
  aucun code : les runs du notebook vont dans `notebooks/mlruns/`, seul
  backend réellement utilisé.
- **Test qui polluait le dépôt corrigé.** En cherchant d'où revenait ce
  `mlflow.db`, il est apparu que `test_load_model_file_not_found` interrogeait
  le **vrai** registry MLflow : `load_model()` y bascule en dernier recours, et
  ce test lui passait un chemin inexistant sans simuler MLflow. Chaque
  exécution de la suite recréait donc une base dans le dépôt et attendait un
  service distant. MLflow y est maintenant simulé. Au passage,
  `test_load_model_with_mlflow_uri_env` ne contenait aucune assertion — il
  passait à vide ; réécrit, il couvre réellement le repli MLflow, ce qui fait
  passer `src/model.py` de 78 % à 86 % de couverture.
- **Figures renommées** en minuscules ASCII, séparées par underscores. Les
  anciens noms mélangeaient majuscules, accents (`corrélation_…`, gênant sur
  d'autres systèmes de fichiers) et abréviations opaques : `Distribution_V_C`
  et `Distribution_V_N` sont devenus `distribution_categorical_features` et
  `distribution_numerical_features`.
- **Documentation ramenée à trois fichiers** : `README.md` (présentation),
  `REPORT.md` (ce document) et `DEPLOYMENT.md` (production). Les neuf
  fichiers précédents se répétaient et se contredisaient sur les chiffres.
  Le README ne duplique plus la CI, la couverture ni le guide de
  contribution : il y renvoie.
- **Artefacts générés retirés du répertoire de travail** (`htmlcov/`,
  `.pytest_cache/`, `.coverage`, `__pycache__/`) — tous régénérables.
- `LICENSE` (MIT) ajouté : le badge du README annonçait cette licence sans
  qu'aucun fichier ne l'accompagne.
- `.dockerignore` corrigé : la ligne `.github/workflows/  # commentaire`
  était interprétée comme un motif incluant le commentaire, ce format
  n'acceptant pas de commentaire en fin de ligne.

### 3.9 Audit méthodologique des notebooks d'analyse

Une relecture de `01_eda` et `02_processing` a mis au jour des erreurs de
méthode qui avaient faussé des conclusions publiées.

**La corrélation de Pearson servait de seul critère de sélection.** Elle ne
détecte que les relations linéaires : `products_number`, dont le taux de churn
dessine un U (1 produit → 27,7 %, 2 → 7,6 %, 3 → 82,7 %, 4 → 100 %), affichait
une corrélation de −0,048 et avait été classée parmi les variables faibles.
Son information mutuelle (0,070) en fait pourtant **la variable la plus
informative du jeu de données**. L'AUC univariée (0,42) échoue de la même
façon, pour la même raison : elle suppose une relation monotone. L'EDA mesure
désormais l'information mutuelle et affiche les taux de churn par modalité.

**Trois variables sans signal n'avaient pas été identifiées** :
`estimated_salary` suit une loi uniforme (Kolmogorov-Smirnov, p = 0,84 :
indiscernable d'un tirage au hasard), et `credit_score` comme `tenure` ont une
information mutuelle nulle ou quasi nulle. `tenure_group` a pourtant été
construite sur `tenure`.

**L'interprétabilité publiée était fausse.** Le README et le notebook 03
présentaient `has_balance`, `age_group_very_old` et `tenure_group_loyal`
comme les trois variables dominantes selon SHAP. La cellule de calcul du
notebook imprime en réalité `['products_number', 'age', 'gender_Male']` : les
trois variables annoncées occupent les rangs **9, 17 et 19 sur 19**,
`tenure_group_loyal` étant la moins utilisée du modèle. Le sens de l'effet de
`has_balance` était en outre inversé — les clients à solde nul churnent moins,
pas plus. Les trois documents ont été corrigés à partir de la sortie réelle.

**Un effet confondu n'avait jamais été testé** : aucun client allemand n'a un
solde nul, et l'Allemagne churne à 32,4 %. `has_balance` mesurait donc en
partie le pays. Le croisement manquant a été ajouté à l'EDA.

**La détection d'outliers concluait à l'envers.** L'écart interquartile
signalait 359 clients de 63 à 92 ans comme « hors normes » alors qu'ils
churnent à 20,3 %, soit exactement la moyenne. À l'inverse, les 15
`credit_score` les plus bas et les 60 clients détenant 4 produits churnent à
**100 %** : l'IQR désignait les lignes les plus informatives du jeu de
données. L'analyse compare désormais le comportement des points extrêmes à
celui de la population et conclut explicitement — anomalie ou signal — au lieu
de se contenter de les compter. Aucune valeur n'est supprimée.

**Deux défauts techniques corrigés dans `01_eda`** :
`warnings.filterwarnings('ignore')` masquait tous les avertissements, y
compris ceux signalant un vrai problème ; et une cellule mutait une liste
définie plus haut (`numerical_cols.remove('churn')`), ce qui empêchait de la
ré-exécuter seule. La liste des variables explicatives est maintenant
construite une fois, sans mutation, et l'appel `seaborn` déprécié qu'occultait
le filtre a été mis à jour.

### 3.10 Encodage des produits, métriques et seuil de décision

**`products_number` est encodé selon sa forme réelle.** Deux modalités
(`products_2`, `products_3_plus`, référence « 1 produit ») rendent explicite
la relation en U. Les niveaux 3 et 4 sont regroupés : ils se comportent de la
même façon et 4 produits ne concerne que 60 clients, trop peu pour une
indicatrice stable. Le contrat passe de 19 à 21 features.

Le gain n'est pas là où on l'attendait. LightGBM, qui reconstruisait déjà la
forme en U par découpages successifs, ne progresse pas. En revanche la
**régression logistique gagne 5 points d'AUC-ROC (0.795 → 0.848)** et devient
deuxième au PR-AUC : un modèle linéaire ne pouvait pas exprimer cette relation
avec un coefficient unique, et les modalités la lui donnent.

Cette étape a par ailleurs révélé que le nombre **19 était codé en dur à onze
endroits**, dont `predict_churn`, `validate_model` et six tests. Tout est
maintenant dérivé de `len(FEATURE_COLUMNS)` : un futur changement de features
ne passera plus inaperçu.

**Le PR-AUC devient la métrique de référence.** Avec 20,4 % de positifs,
l'AUC-ROC intègre un taux de vrais négatifs gonflé par la classe majoritaire.
Sa référence aléatoire est 0.5 quand celle du PR-AUC est le taux de churn
lui-même (0.204) — le second discrimine bien mieux entre modèles. Le tuning
optimise désormais `average_precision`, ce qui, mesures à l'appui, n'a rien
changé aux hyperparamètres sélectionnés : la métrique de tuning importe moins
ici que la métrique de lecture.

**`class_weight='balanced'` est retiré du modèle déployé.** Ce paramètre
déformait les probabilités : le modèle surestimait le risque d'un facteur 1,8,
si bien qu'un client affiché « à 70 % » churnait en réalité moins d'une fois
sur deux. Comme la décision repose sur un seuil calibré sur le coût et non sur
le 0,5 implicite, ce rééquilibrage ne servait à rien et coûtait la lisibilité.

| Probabilité affichée | Churn réel — avant | Churn réel — après |
|---|---|---|
| 40–60 % | 19,0 % | 48,0 % |
| 60–80 % | ~40 % | 82,3 % |

Le score de Brier passe de **0.1370 à 0.0993 (−27 %)**, à pouvoir
discriminant inchangé. Il est conservé pour le benchmark des huit modèles, où
la comparaison doit rester à armes égales.

**Le seuil de décision est choisi hors échantillon.** Il était optimisé
directement sur le jeu de test, ce qui produisait un coût flatteur mais
irréproductible. Il est désormais déterminé par validation croisée sur le
train, puis appliqué au test : 0.10, pour un coût de 54 750 € contre 108 500 €
au seuil naïf de 0,50. Il est stocké dans `model_info.pkl` et lu à l'exécution
par l'application via `get_decision_threshold()`, afin qu'il ne puisse pas
diverger du modèle servi.

**Deux sources de valeurs recopiées à la main ont été supprimées** : le
graphique comparatif des modèles et `model_info.pkl` contenaient des métriques
en dur (`auc_roc: 0.8715`). Elles sont maintenant mesurées à l'exécution,
donc impossibles à désynchroniser du modèle sauvegardé.

**Ce qui n'a pas progressé, et pourquoi.** Le PR-AUC reste à 0.719 et le coût
métier ne baisse pas. Une mesure par bootstrap sur les 2 000 clients de test
donne un intervalle de confiance à 95 % large de ±0.04 pour le PR-AUC et
±0.02 pour l'AUC-ROC — soit dix fois les écarts observés entre les variantes
testées. Le jeu de données plafonne : les gains réels de cette étape portent
sur la calibration et sur la validité méthodologique, pas sur la
discrimination.

### 3.11 Cibler par valeur attendue plutôt que par probabilité

Une fois le modèle au plafond de ce que les données permettent, le gain se
déplace de la prédiction vers la décision. Le seuil unique (contacter si
p ≥ 0.10) traite tous les clients comme s'ils valaient la même chose. Or
sauver un client à 0 € de solde ne rapporte pas autant que sauver un client à
200 000 €.

La règle appliquée est celle de la valeur attendue : **agir tant que
`probabilité × valeur du client` dépasse le coût du contact**. Le seuil n'est
plus fixe, il s'adapte à chaque client.

**Condition de pertinence, vérifiée avant d'implémenter.** L'approche
n'apporte rien si valeur et risque se recouvrent. Ici la corrélation entre les
deux n'est que de +0,218, et la valeur moyenne est plate sur les trois
quartiles de risque les plus élevés (569, 581, 563 €) : parmi les clients à
risque, la valeur varie indépendamment.

| Politique (2 000 clients de test) | Coût | Clients contactés |
|---|---|---|
| Ne contacter personne | 224 808 € | 0 |
| Contacter tout le monde | 100 000 € | 2 000 |
| Seuil sur la probabilité (0.10) | 74 235 € | 1 051 |
| **Valeur attendue** | **70 574 €** | 1 101 |

**Gain mesuré : 3 661 €, soit 4,9 %, sans réentraînement** — de l'ordre de
18 000 € rapporté aux 10 000 clients. Le gain reste positif sur toute la plage
d'hypothèses de valeur testée (+1 448 € à +8 005 €), il ne dépend donc pas du
réglage exact.

Le déplacement est interprétable : 151 clients à 8,4 % de risque mais 747 € de
valeur entrent dans le ciblage, tandis que 108 clients à 16,2 % de risque et
205 € de valeur en sortent — les sauver rapporterait 33 €, les contacter en
coûte 50.

> **Hypothèse assumée.** La valeur client (`base + part du solde`) est un
> paramètre de travail défini dans `config.py`, pas une mesure. Il doit être
> remplacé par le modèle de valeur vie client de la banque. La règle de
> décision, elle, reste valable quel que soit ce modèle.

---

## 4. Architecture

### Flux de données

```
data/raw/Bank-Customer-Churn-Prediction.csv     (dataset Kaggle, non versionné)
        │
        │  notebooks/01_eda.ipynb          exploration, distributions, outliers
        ▼
  build_features_frame()                   src/features.py
        │                                  encodage + features dérivées
        │  notebooks/02_processing.ipynb
        ▼
data/processed/churn_dataset_clean.csv     10 000 × 22 (21 features + churn)
        │
        │  notebooks/03_modeling.ipynb     benchmark 8 modèles, tuning, SHAP
        ▼
models/best_model.pkl                      LightGBM tuné (AUC-ROC 0.87)
models/model_info.pkl                      métriques + hyperparamètres
        │
        ▼
streamlit_app.py                           saisie d'un client → probabilité
        │                                  + explication SHAP top 3
        ▼
logs/predictions.csv                       journal des prédictions (drift)
```

### Modules `src/`

| Module | Responsabilité |
|--------|----------------|
| `config.py` | Source unique des chemins, seuils métier et `RANDOM_STATE`. Aucun autre fichier ne code de constante en dur. |
| `features.py` | Feature engineering : `build_features()` pour un client, `build_features_frame()` pour le dataset complet. |
| `validation.py` | Schéma Pydantic des entrées client (plages, énumérations) avant tout calcul. |
| `model.py` | Chargement (fichier local, puis MLflow en repli), contrôle d'intégrité, prédiction. |
| `explainer.py` | Enveloppe SHAP `TreeExplainer` avec cache et extraction des facteurs dominants. |
| `logger.py` | Logging centralisé, rotation quotidienne, handlers partagés. |
| `prediction_logger.py` | Journalisation CSV des prédictions servies. |
| `decision.py` | Traduit une probabilité en action : contacter ou non, et pour quel gain. |
| `exceptions.py` | Hiérarchie d'exceptions, toutes sous `ChurnPredictionError`. |

### Choix structurants

**Pourquoi les notebooks importent-ils `src/` plutôt que l'inverse ?**
Les notebooks racontent l'analyse ; le code réutilisable vit dans `src/`, qui
est testé et importé aussi bien par l'application que par les notebooks. Un
notebook ne peut pas être testé automatiquement, `src/` si.

**Pourquoi conserver deux fonctions de feature engineering ?**
Les deux contextes diffèrent réellement : l'entraînement transforme 10 000
lignes vectorisées, le service transforme un dictionnaire de formulaire. Les
fusionner obligerait le chemin de service à fabriquer un DataFrame brut à
chaque appel. Les tests de parité offrent la même garantie sans ce détour.

**Pourquoi un CSV de prédictions plutôt que MLflow ?**
La détection de drift a besoin des entrées servies, pas des métriques
d'entraînement. Un CSV en append reste lisible sans infrastructure et se
charge directement dans pandas.

---

## 5. Données

Les fichiers `.csv` ne sont pas versionnés. Source :
[Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
(`gauravtopre/bank-customer-churn-dataset`). Téléchargez-le, placez-le dans
`data/raw/Bank-Customer-Churn-Prediction.csv`, puis exécutez
`notebooks/02_processing.ipynb` pour régénérer le dataset traité.

### Dataset brut

10 000 clients, 12 colonnes, aucune valeur manquante, aucun doublon.

| Colonne | Type | Description | Plage |
|---------|------|-------------|-------|
| `customer_id` | int | Identifiant client | retiré au traitement |
| `credit_score` | int | Score de crédit | 350 – 850 |
| `country` | str | Pays | France, Germany, Spain |
| `gender` | str | Genre | Female, Male |
| `age` | int | Âge | 18 – 92 |
| `tenure` | int | Ancienneté bancaire (années) | 0 – 10 |
| `balance` | float | Solde du compte (€) | 0 – 250 898 |
| `products_number` | int | Produits détenus | 1 – 4 |
| `credit_card` | int | Détient une carte | 0 / 1 |
| `active_member` | int | Membre actif | 0 / 1 |
| `estimated_salary` | float | Salaire estimé (€) | 12 – 199 992 |
| `churn` | int | **Cible** : le client est parti | 0 / 1 |

**Déséquilibre de classes** : 20,4 % de churn. C'est ce qui justifie le split
stratifié, le `class_weight='balanced'` des modèles, et le choix de l'AUC-ROC
plutôt que de l'accuracy comme métrique de sélection.

### Features dérivées

| Feature | Construction | Hypothèse métier initiale | Vérification |
|---------|--------------|---------------------------|--------------|
| `age_group_*` | Tranches 30 / 40 / 50 / 60 ans | Le comportement de churn varie par génération | Confirmée : 6,3 % de churn avant 30 ans, 58,6 % entre 50 et 60 ans, repli au-delà |
| `has_balance` | `balance > 0` | Un client sans solde actif est moins engagé | **Infirmée** : les clients à solde nul churnent *moins* (13,8 % contre 24,1 %), et l'écart est en partie un effet pays |
| `tenure_group_*` | Tranches 2 / 5 ans | La fidélité n'évolue pas linéairement | **Non étayée** : `tenure` a une information mutuelle de 0,0007, et `tenure_group_loyal` est la variable la moins utilisée du modèle (rang 19/19) |
| `active_products` | `active_member × products_number` | Combine deux signaux faibles en un signal composite | Partiellement utile, mais `products_number` seul est bien plus informatif |
| `products_2`, `products_3_plus` | Modalités du nombre de produits | La relation avec le churn est en U, pas linéaire | Confirmée : 8,4 % de churn à 2 produits contre 30,5 % ailleurs. `products_2` est la 2ᵉ feature du modèle |

**Effet confondu sur `has_balance`** : aucun client allemand n'a un solde nul,
alors que l'Allemagne est le pays qui churne le plus (32,4 % contre ~16 %).
`has_balance = 0` implique donc « France ou Espagne » dans 100 % des cas.
À pays constant, l'écart tombe de +10,3 points à +4,3 (France) et +6,0
(Espagne). La variable garde un signal propre, mais deux fois plus faible que
ne le suggère la lecture brute.

**Multicolinéarité** : certaines dérivées sont fortement corrélées à leur
variable d'origine (`has_balance`/`balance` : 0,92). Sans effet sur les
modèles à base d'arbres retenus ; les modèles linéaires du benchmark sont
entraînés sans ces colonnes (`balance`, `tenure`, `active_products` retirées).

---

## 6. Chaîne de qualité

```
pre-commit  →  black + flake8 (avant chaque commit, en local)
     │
     ▼
GitHub Actions .github/workflows/tests.yml     (push et PR sur main/develop)
     ├─ lint      black --check + flake8
     ├─ test      pytest + couverture, seuil 80 %
     └─ security  bandit sur src/
     │
     ▼ (si succès, sur main)
deploy.yml  →  vérifie que streamlit_app.py compile
     │
     ▼
Streamlit Cloud  →  redéploiement automatique depuis main
```

`pyproject.toml` (black, pytest, coverage) et `.flake8` sont partagés entre le
poste local et la CI : les deux environnements appliquent exactement les mêmes
règles, ce qui évite les échecs « ça passe chez moi ».

### Reproduire la CI en local

```bash
black --check src/ tests/ streamlit_app.py
flake8 src/ tests/ streamlit_app.py
pytest tests/ --cov=src --cov-report=term-missing
bandit -r src/ --skip B101
```

Pour automatiser ces contrôles avant chaque commit :

```bash
pip install pre-commit && pre-commit install
```

### Couverture actuelle

```
src/config.py           : 100%
src/features.py         : 100%
src/exceptions.py       : 100%
src/validation.py       : 100%
src/logger.py           : 100%
src/model.py            :  86%   erreurs de lecture du pickle
src/explainer.py        :  74%   cas limites SHAP multiclasse
src/prediction_logger.py:  74%   erreurs d'écriture CSV
──────────────────────────────
TOTAL                   :  88%
```

Les zones non couvertes sont des branches d'erreur d'infrastructure,
difficiles à provoquer sans monter un serveur MLflow ou simuler un disque
plein.

---

## 7. Contribuer

### Mise en place

```bash
git clone https://github.com/EddieZIDA/churn-prediction.git
cd churn-prediction
python -m venv venv
venv/Scripts/activate        # Linux/macOS : source venv/bin/activate
pip install -r requirements.txt
pre-commit install
pytest tests/                # doit afficher 123 passed
```

### Cycle de contribution

1. Créer une branche : `feature/…`, `fix/…`, `docs/…`, `test/…`, `refactor/…`
2. Développer, avec un test qui échoue avant le correctif et passe après
3. Valider en local : `pytest tests/ --cov=src` puis `black` et `flake8`
4. Committer avec un message typé (`feat:`, `fix:`, `docs:`, `test:`,
   `refactor:`, `chore:`) expliquant le **pourquoi**
5. Ouvrir une PR : la CI doit être verte pour permettre le merge

### Règles du projet

- **Aucune constante en dur** hors de `config.py` — seuils, chemins, graines.
- **Toucher au feature engineering implique de faire passer les tests de
  parité.** Si une transformation change, elle doit changer pour les deux
  chemins à la fois.
- **Exceptions typées uniquement** : lever une classe de `exceptions.py`,
  jamais `Exception`, et chaîner la cause avec `raise … from e`.
- **Logging via `get_logger(__name__)`**, jamais `print()` ni
  `logging.basicConfig()`.
- La couverture ne doit pas descendre sous 80 %, seuil appliqué par la CI.

---

## 8. Ce qui reste

| Sujet | Pourquoi |
|-------|----------|
| Détection de drift | `logs/predictions.csv` collecte les données ; l'analyse comparant les distributions servies à celles d'entraînement reste à écrire. |
| Pipeline de réentraînement | Aujourd'hui manuel via le notebook 03. Une orchestration le rendrait reproductible et planifiable. |
| API batch | L'app traite un client à la fois ; une API permettrait de scorer un portefeuille entier. |
| Artefacts MLflow versionnés | `notebooks/mlruns/` est suivi par git (artefacts et modèle loggé). À déplacer vers un stockage d'artefacts si le dépôt s'alourdit. |
| Couverture des branches MLflow | Nécessite un serveur de test ou des mocks plus poussés. |
