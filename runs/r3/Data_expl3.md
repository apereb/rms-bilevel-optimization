# Analyse de sensibilité — RMS bi-level, `instance_period3`

Run de référence : `runs/r3/baseline/period3_baseline.meta.json` (mode `full`,
aucun appel Gurobi supplémentaire dans cette analyse).

**Univers de configurations feasibles : 4500** (confirmé par la table du front
de Pareto : "15 config(s) out of 4500").

ε=0.15 (corridor_size=260, identique dans `epsilon-sweep` à ε=0.15,
`tau-sweep` et `alpha-sweep` à α=5).

---

## 1. Sensibilité à ε (`epsilon-sweep`)

```
python param_sweep.py epsilon-sweep instances/instance_period3.json \
    runs/r3/baseline/period3_baseline.meta.json \
    --epsilons 0.05,0.1,0.15,0.2,0.3,0.4
```

C* = 31.2548, indépendant de ε.

| ε | C* | threshold | corridor_size | winner_idx | winner_C_config | winner_Q | winner_Phi |
|---|---|---|---|---|---|---|---|
| 0.05 | 31.2548 | 32.8175 | 26 | 1 | 28.10 | 19.34 | 4.2615 |
| 0.1 | 31.2548 | 34.3802 | 99 | 45 | 29.25 | 19.34 | 5.1830 |
| 0.15 | 31.2548 | 35.9430 | 260 | 90 | 31.25 | 19.34 | 6.6552 |
| 0.2 | 31.2548 | 37.5057 | 550 | 1080 | 32.90 | 19.34 | 7.1115 |
| 0.3 | 31.2548 | 40.6312 | 1432 | 195 | 35.33 | 19.21 | 9.0607 |
| 0.4 | 31.2548 | 43.7567 | 2590 | 199 | 38.78 | 18.88 | 9.8385 |

**Observation.** Corridor de 26 configs (0.6 % de l'univers) à ε=0.05,
à 2590 (57.6 %) à ε=0.4 — croissance comparable en proportion à
`period2`. À ε=0.15 (défaut), 260 configs, soit 5.8 % de l'univers : le
plus sélectif des trois instances au même ε (`period1` : 11.6 %,
`period2` : 7.9 %, `period3` : 5.8 %).

`winner_Q` reste plat à 19.34 de ε=0.05 à ε=0.2 (4 valeurs identiques),
puis baisse seulement aux deux dernières valeurs (19.21 à ε=0.3, 18.88 à
ε=0.4) — un plateau plus large qu'observé sur `period1` ou `period2`.

---

## 2. Sensibilité à τ (`tau-sweep`)

```
python param_sweep.py tau-sweep instances/instance_period3.json \
    runs/r3/baseline/period3_baseline.meta.json \
    --taus 1,2,3,5,8,13,21,50,100
```

Corridor constant (260 configurations) sur tout le balayage.

### 2.1 Gagnant du corridor vs. τ

| τ | corridor_size | winner_idx | winner_C_config | winner_Q | winner_Theta | winner_Phi |
|---|---|---|---|---|---|---|
| 1 | 260 | 90 | 31.25 | 19.34 | 6.4330 | 5.4330 |
| 2 | 260 | 105 | 31.33 | 19.21 | 6.3832 | 5.8263 |
| 3 | 260 | 90 | 31.25 | 19.34 | 6.4330 | 6.6552 |
| 5 | 260 | 90 | 31.25 | 19.34 | 6.4330 | 6.5187 |
| 8 | 260 | 4 | 30.70 | 18.45 | 3.3358 | 7.4204 |
| 13 | 260 | 22 | 29.82 | 18.88 | 2.7824 | 7.9987 |
| 21 | 260 | 2253 | 30.55 | 18.69 | 2.7348 | 7.4037 |
| 50 | 260 | 22 | 29.82 | 18.88 | 2.7824 | 8.0487 |
| 100 | 260 | 22 | 29.82 | 18.88 | 2.7824 | 8.0487 |

**Deux points inhabituels par rapport à `period1`/`period2` :**

1. **Le gagnant repasse deux fois par `#90`** (τ=1, puis τ=3 et τ=5,
   après être passé par `#105` à τ=2) avant de diverger définitivement à
   τ=8. Ce n'est pas un simple cheminement monotone vers le gagnant
   final.
2. **`#2253` à τ=21 est un pic isolé, pas une étape vers la
   convergence** : son Φ (7.4037) est *inférieur* à celui du gagnant à
   τ=13 (`#22`, Φ=7.9987) et à τ=50 (`#22`, Φ=8.0487). Autrement dit,
   à τ=21, un autre candidat dépasse temporairement `#22` en Φ, avant que
   `#22` ne reprenne l'avantage à τ=50 — le Φ du gagnant n'est donc pas
   monotone croissant en τ, y compris pour le gagnant global final.

**Stabilisation : τ=50** (comme `period2`, pas τ=13 comme `period1`) —
`#22` est gagnant à τ=13, perd temporairement l'avantage à τ=21, puis
regagne et se stabilise à τ=50 et 100 (Φ identique : 8.0487).

### 2.2 Distribution de la taille de voisinage sur l'univers complet

| τ | min | p25 | médiane | moyenne | p75 | p90 | max | frac_isolés |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 2 | 1.87 | 2 | 3 | 3 | 3.6% |
| 2 | 2 | 5 | 6 | 5.90 | 7 | 9 | 10 | 0.0% |
| 3 | 5 | 9 | 12 | 12.30 | 15 | 18 | 21 | 0.0% |
| 5 | 8 | 17 | 22 | 22.94 | 29 | 34 | 38 | 0.0% |
| 8 | 13 | 23 | 35 | 33.63 | 39 | 50 | 63 | 0.0% |
| 13 | 14 | 29 | 39 | 41.80 | 55 | 75 | 79 | 0.0% |
| 21 | 26 | 57 | 74 | 75.28 | 92 | 124 | 138 | 0.0% |
| 50 | 29 | 59 | 79 | 87.67 | 119 | 159 | 159 | 0.0% |
| 100 | 29 | 59 | 79 | 87.67 | 119 | 159 | 159 | 0.0% |

Saturation structurelle entre τ=50 et τ=100 (identique : moyenne 87.67,
p90/max=119/159 fixes) — **troisième instance sur trois où la
stabilisation du gagnant coïncide avec τ=50**, pas avec un τ intermédiaire
plus petit. Ceci renforce (section 8) le constat déjà fait sur `period2` :
**la stabilisation précoce observée sur `period1` (τ=13) semble être
l'exception, pas la règle**, sur cet ensemble de 3 instances.

---

## 3. Sensibilité à `phi_top_fraction` (`phi-fraction-sweep`, simulation)

```
python param_sweep.py phi-fraction-sweep instances/instance_period3.json \
    runs/r3/baseline/period3_baseline.meta.json \
    --fractions 0.1,0.2,0.3,0.5,0.7,1.0
```

Gagnant de référence (fraction=100 %) : **`config_idx=90`**.

| fraction | kept_n | C* (simulé) | corridor_size | winner_idx | winner==baseline | écart de coût vs vrai C* |
|---|---|---|---|---|---|---|
| 10% | 450 | 39.8548 | 123 | 200 | False | +44.86% |
| 20% | 900 | 37.8548 | 150 | 199 | False | +38.46% |
| 30% | 1350 | 37.1738 | 210 | 202 | False | +35.39% |
| 50% | 2250 | 35.7638 | 290 | 196 | False | +30.39% |
| 70% | 3150 | 33.8548 | 234 | 151 | False | +23.99% |
| 100% | 4500 | 31.2548 | 260 | 90 | True | +14.72%¹ |

¹ Écart normal Phase 1/Phase 2 à 100 %.

**C'est la pire des trois instances pour `partial-inversion`** : le vrai
gagnant n'est retrouvé à **aucune** des fractions testées avant 100 %
(comme `period1`), et les écarts de coût sont les plus élevés observés
(+44.86 % à 10 %, contre +32.42 % sur `period1` et +32.42 % sur `period2`
à la même fraction). Même à 70 %, l'écart reste de +23.99 %.

---

## 4. Recall structurel du corridor (`heuristic_analysis.py phi-recall`)

```
python heuristic_analysis.py phi-recall instances/instance_period3.json \
    runs/r3/baseline/period3_baseline.meta.json \
    --fractions 0.1,0.2,0.3,0.5,0.7,1.0
```

Baseline : C* = 31.2548, corridor size = 260, winner config_idx = 90.

| fraction | kept_n | corridor_recall | winner_included |
|---|---|---|---|
| 10% | 450 | 0.0% | False |
| 20% | 900 | 0.0% | False |
| 30% | 1350 | 0.0% | False |
| 50% | 2250 | 1.5% | True |
| 70% | 3150 | 9.6% | True |
| 100% | 4500 | 100.0% | True |

**Nettement plus dégradé que les deux instances précédentes** : le
gagnant n'est même pas dans le sous-univers gardé avant 50 % (contre 20 %
sur `period1` et `period2`), et le recall du corridor à 70 % n'est que de
9.6 % (contre 28.9 % sur `period1` et 41.3 % sur `period2`). Cohérent
avec les écarts de coût plus sévères observés en section 3.

---

## 5. Sensibilité à α — Phase 1 vs. Phase 2 (`alpha-sweep`)

```
python heuristic_analysis.py alpha-sweep instances/instance_period3.json \
    runs/r3/baseline/period3_baseline.meta.json \
    --alphas 1,2,...,20 --top-k 20
```

Reference alpha = 5, C* = 31.2548, corridor_size = 260, winner = 90 —
cohérent avec les sections 3 et 4.

### 5.1 Table complète

| α | C* | corridor_size | winner_idx (P2) | winner_C_config | winner_Q | phase1_winner_idx | phase1_C_config | phase1_Q | P1==P2 | topK_jaccard | topK_spearman | topK_kendall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 27.5710 | 192 | 48 | 30.70 | 18.79 | 225 | 26.65 | 19.34 | **False** | 0.739 | 0.988 | 0.947 |
| 2 | 28.4919 | 208 | 48 | 30.70 | 18.79 | 225 | 26.65 | 19.34 | **False** | 0.739 | 0.988 | 0.947 |
| 3 | 29.4129 | 224 | 48 | 30.70 | 18.79 | 225 | 26.65 | 19.34 | **False** | 0.818 | 0.988 | 0.947 |
| 4 | 30.3338 | 244 | 993 | 31.29 | 18.79 | 225 | 26.65 | 19.34 | **False** | 0.905 | 0.995 | 0.979 |
| 5 | 31.2548 | 260 | 90 | 31.25 | 19.34 | 225 | 26.65 | 19.34 | **False** | 1.000 | 1.000 | 1.000 |
| 6 | 32.1757 | 281 | 90 | 31.25 | 19.34 | 225 | 26.65 | 19.34 | **False** | 1.000 | 0.997 | 0.979 |
| 7 | 33.0967 | 311 | 90 | 31.25 | 19.34 | 225 | 26.65 | 19.34 | **False** | 1.000 | 0.992 | 0.968 |
| 8 | 34.0176 | 335 | 90 | 31.25 | 19.34 | 225 | 26.65 | 19.34 | **False** | 1.000 | 0.992 | 0.968 |
| 9 | 34.9386 | 355 | 90 | 31.25 | 19.34 | 225 | 26.65 | 19.34 | **False** | 1.000 | 0.992 | 0.968 |
| 10 | 35.8595 | 381 | 90 | 31.25 | 19.34 | 225 | 26.65 | 19.34 | **False** | 0.905 | 0.991 | 0.958 |
| 11 | 36.7805 | 416 | 90 | 31.25 | 19.34 | 225 | 26.65 | 19.34 | **False** | 0.905 | 0.991 | 0.958 |
| 12 | 37.7014 | 436 | 90 | 31.25 | 19.34 | 225 | 26.65 | 19.34 | **False** | 0.905 | 0.988 | 0.947 |
| 13 | 38.6219 | 469 | 93 | 32.70 | 18.79 | 240 | 26.73 | 19.21 | **False** | 0.905 | 0.962 | 0.842 |
| 14 | 39.5367 | 508 | 93 | 32.70 | 18.79 | 240 | 26.73 | 19.21 | **False** | 0.818 | 0.962 | 0.842 |
| 15 | 40.4514 | 541 | 93 | 32.70 | 18.79 | 240 | 26.73 | 19.21 | **False** | 0.818 | 0.962 | 0.842 |
| 16 | 41.3662 | 579 | 93 | 32.70 | 18.79 | 240 | 26.73 | 19.21 | **False** | 0.818 | 0.962 | 0.842 |
| 17 | 42.2810 | 598 | 1095 | 32.98 | 19.21 | 240 | 26.73 | 19.21 | **False** | 0.739 | 0.958 | 0.832 |
| 18 | 43.1957 | 639 | 1095 | 32.98 | 19.21 | 240 | 26.73 | 19.21 | **False** | 0.739 | 0.949 | 0.821 |
| 19 | 44.1105 | 685 | 150 | 33.33 | 19.21 | 240 | 26.73 | 19.21 | **False** | 0.739 | 0.943 | 0.811 |
| 20 | 45.0252 | 721 | 150 | 33.33 | 19.21 | 240 | 26.73 | 19.21 | **False** | 0.739 | 0.935 | 0.800 |

**`P1==P2` reste `False` sur les 20 valeurs testées — troisième instance
sur trois, aucune exception.**

### 5.2 Trajectoire du gagnant Phase 1

Seulement **2 gagnants distincts**, comme `period2` :

| Plage d'α | phase1_winner_idx | C_config | Q |
|---|---|---|---|
| [1, 12] | 225 | 26.65 | 19.34 |
| [13, 20] | 240 | 26.73 | 19.21 |

Bascule unique, entre α=12 et α=13.

### 5.3 Intervalles de stabilité du gagnant Phase 2

| winner_idx | plage d'α | largeur |
|---|---|---|
| 48 | [1, 3] | 3 |
| 993 | [4, 4] | 1 |
| **90** | **[5, 12]** | **8** |
| 93 | [13, 16] | 4 |
| 1095 | [17, 18] | 2 |
| 150 | [19, 20] | 2 |

**Le plateau le plus large observé sur les trois instances** : `#90`
reste gagnant sur 8 valeurs d'α consécutives ([5,12]) — plus large que
`#237` sur `period2` ([15,20], largeur 6) et très supérieur à tout ce qui
a été vu sur `period1` (largeur max 2). Le gagnant à α=5 (l'α de
référence) est donc, sur cette instance, un choix particulièrement
robuste.

Cohérent avec ce plateau large : **`topK_jaccard` reste à 1.000 sur
α=5 à 9 inclus** (5 valeurs consécutives) — le classement des 20
meilleures configurations est totalement figé sur cette plage, du jamais
vu sur les deux instances précédentes (où seule la valeur de référence
elle-même atteignait 1.000).

---

## 6. Visualisation Pareto (`pareto3.png`)

```
python heuristic_analysis.py pareto instances/instance_period3.json \
    runs/r3/baseline/period3_baseline.meta.json \
    --alphas 1,2,...,20 --output-png pareto3.png
```

Front de Pareto (**15 configurations non dominées sur (C_config, Q)**,
soit 0.3 % de l'univers) :

| config_idx | C_config | Q |
|---|---|---|
| 225 | 26.65 | 19.34 |
| 240 | 26.73 | 19.21 |
| 241 | 27.58 | 19.21 |
| 228 | 28.10 | 19.01 |
| 243 | 28.18 | 18.88 |
| 903 | 28.23 | 18.79 |
| 902 | 28.48 | 18.65 |
| 2 | 28.95 | 18.15 |
| 2252 | 30.80 | 18.05 |
| 1352 | 31.80 | 17.86 |
| 2027 | 31.93 | 17.86 |
| 1127 | 32.40 | 17.00 |
| 3422 | 36.25 | 16.90 |
| 3497 | 40.85 | 16.90 |
| 3479 | 43.40 | 16.90 |

Le plot confirme bien les deux tables ci-dessus : **2 losanges orange**
distincts (`#225` à α=1, `#240` apparaissant plus bas — première
occurrence à α=13) et **6 points noirs** distincts (`#48`, `#993`, `#90`,
`#93`, `#1095`, `#150`). Les 6 points noirs se répartissent en 3
groupes visuellement : `#90` isolé en haut (Q=19.34, seul gagnant à ce
niveau de makespan), un groupe de 3 points serrés autour de Q≈18.79
(`#48`, `#993`, `#93` — C_config 30.70 à 32.70), et un groupe de 2 points
serrés autour de Q≈19.21 (`#1095`, `#150` — C_config 32.98 et 33.33).
Cette proximité fait se chevaucher les étiquettes texte sur le graphe
(ex. "a=1/a=4/a=13" illisibles séparément) — un effet cosmétique du fait
que ces gagnants sont très proches en coût/makespan, pas une erreur de
tracé.

---

## 7. Étude qualitative

### 7.1 Gagnant Phase 2 à α=5 (`#90`) vs. gagnant Phase 1 sur [1,12] (`#225`)

```bash
python config_diff.py instances/instance_period3.json \
    --run runs/r3/baseline/period3_baseline.meta.json --ids 90,225
```

**4 des 5 étages sont strictement identiques** (stages 0, 1, 3, 4 :
même `machine_id`, même `config_id`, même `n`) — seul l'étage 2 diffère,
et il diffère complètement : type de machine (2→3), configuration (3→1),
nombre d'unités (3→1), modules entièrement renouvelés (ajout de
[10,13,16], retrait de [23], seuls [15,19] partagés). C_config passe de
31.25 à 26.65 (Δ=-4.60), entièrement porté par ce changement d'un seul
étage.

**Malgré ce recouvrement de 4/5 étages, `T_rec = +∞` dans les deux
sens** : le modèle ne fait aucune distinction entre "4 étages identiques,
1 étage complètement différent" et "tous les étages différents" — dès
qu'un seul étage change de type de machine, la paire devient
inatteignable, point. C'est un exemple concret et frappant pour le
mémoire : deux configurations de coût proche (26.65 vs 31.25) et
structurellement quasi-identiques n'apportent *aucune* contribution
mutuelle à Φ, uniquement parce que l'étage qui diffère implique un
changement de type de machine plutôt qu'un simple changement de
configuration ou de nombre d'unités.

### 7.2 L'anomalie du τ-sweep : `#2253` (pic à τ=21) vs. `#22` (gagnant à τ=13/50)

```bash
python config_diff.py instances/instance_period3.json \
    --run runs/r3/baseline/period3_baseline.meta.json --ids 2253,22
```

Ici encore `T_rec = +∞` dans les deux sens, mais pour une raison
différente de 7.1 : **deux étages diffèrent en type de machine** (étage 3 :
2→3 ; étage 4 : 2→3), pas un seul. L'étage 0 diffère aussi, mais
seulement en configuration (`same_type=True`, `same_config=False` :
4→2, modules quasi entièrement renouvelés) — un changement de retooling,
pas de type. C_config est quasiment identique (30.55 vs 29.82,
Δ=-0.73 seulement).

**Ceci confirme le mécanisme de l'anomalie repérée en section 2.1** :
`#2253` (signature machine-type `(2,2,2,2,2)`) et `#22` (signature
`(2,2,2,3,3)`) appartiennent à des *buckets différents* dans
`structural.compute_measures` (le bucketing par signature de type de
machine, section 2 du README) — leurs voisinages respectifs sont donc
calculés sur des ensembles de candidats totalement disjoints, qui
grossissent chacun à leur propre rythme quand τ augmente. Le pic isolé de
`#2253` à τ=21 n'est donc pas une histoire de "reconfiguration entre
`#2253` et `#22`" (ils ne sont jamais reconfiguration-reachable, à
aucun τ) : c'est que le voisinage *propre* à `#2253` (au sein de son
bucket `(2,2,2,2,2)`) a transitoirement gagné en Θ moyen à τ=21 plus vite
que celui de `#22` dans son propre bucket, avant que ce dernier ne
reprenne l'avantage à τ=50. Comme Φ(y) est une moyenne de Θ sur un
ensemble de voisins qui ne fait que croître avec τ, rien ne garantit que
cette moyenne croisse elle-même de façon monotone (un nouveau voisin
inclus à un τ donné peut avoir un Θ inférieur à la moyenne courante et la
tirer vers le bas) — c'est ce mécanisme, combiné au fait que les
compétiteurs évoluent dans des buckets indépendants, qui produit le
croisement non monotone observé.

---

## 8. Comparaison inter-périodes (period1 / period2 / period3)

| | period1 | period2 | period3 |
|---|---|---|---|
| Univers (configs feasibles) | 3024 | 3990 | 4500 |
| Corridor à ε=0.15 (% univers) | 526 (17.4%) | 315 (7.9%) | 260 (5.8%) |
| τ de stabilisation du gagnant | **13** | 50 | 50 |
| τ de saturation structurelle | ~50 | 50 | 50 |
| Gagnant Phase 2 retrouvé par `partial-inversion` à... | jamais (≤70% testé) | 70% | jamais (≤70% testé) |
| Pire écart de coût (fraction=10%) | +31.57% | +32.42% | **+44.86%** |
| P1==P2 sur toute la plage α∈[1,20] | False partout | False partout | False partout |
| Largeur max d'un plateau de gagnant Phase 2 (α) | 2 | 6 | **8** |
| Nb. de gagnants Phase 1 distincts sur α∈[1,20] | 3 | 2 | 2 |

**Constats robustes sur les 3 instances (candidats à devenir des
énoncés généraux dans le mémoire) :**
- La Phase 2 (Φ) modifie *toujours* le choix de la Phase 1 (coût pur),
  quel que soit α testé.
- `partial-inversion` avec le défaut `phi_top_fraction=0.3` n'est fiable
  sur **aucune** des trois instances (écart de coût ≥+22% à cette
  fraction dans les trois cas).

**Constats qui varient d'une instance à l'autre (donc à documenter comme
instance-dépendants, pas comme propriétés du modèle) :**
- Le τ nécessaire pour un gagnant stable : `period1` fait exception
  (τ=13) ; `period2` et `period3` nécessitent d'atteindre la saturation
  structurelle (τ=50).
- La robustesse du gagnant Phase 2 vis-à-vis d'α : très fragile sur
  `period1` (largeur max 2), de plus en plus stable sur `period2` (6) et
  `period3` (8) — pas de tendance monotone évidente en fonction du
  numéro de période à ce stade (seulement 3 points).

---

## 9. Pistes de suite

- Le mécanisme identifié en 7.2 (compétition entre buckets de signature
  machine-type à des rythmes de croissance différents) explique
  l'anomalie sur `period3`, mais on ne sait pas s'il est fréquent ou
  spécifique à cette instance — vérifier sur `period1`/`period2` si le
  gagnant du τ-sweep a également connu un pic non-monotone en cours de
  route (les tableaux collectés ne le montrent pas explicitement, mais
  ça vaut la peine de le vérifier sur des taux intermédiaires plus fins).
- **Si une `period4` est prévue**, reprendre le tableau de la section 8
  pour voir si la tendance de robustesse croissante en α (2 → 6 → 8) se
  confirme ou si `period3` est un plafond/une coïncidence.
- **`heuristic_analysis.py compare`** entre `period1`/`period2`/`period3`
  baselines et de vrais runs `two-pass` (pas seulement `full`), pour voir
  si la variabilité observée ici est spécifique au mode `full` ou se
  retrouve aussi avec `two-pass`.
- Étant donné que **`partial-inversion` n'est fiable sur aucune des trois
  instances** au défaut 0.3, il vaut sans doute la peine de documenter
  explicitement dans le mémoire que cette heuristique n'est pas
  recommandée pour ce type d'instance, plutôt que de la présenter comme
  une option secondaire viable.