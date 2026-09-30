# Analyse de sensibilité — RMS bi-level, `instance_period2`

Run de référence : `runs/r2/baseline/period2_baseline_v2.meta.json` (mode `full`,
aucun appel Gurobi supplémentaire dans cette analyse).

**Univers de configurations feasibles : 3990**

Sauf mention contraire, les paramètres non balayés sont ceux du fichier
d'instance (ε=0.15, confirmé par le corridor_size=315 identique dans
`epsilon-sweep` à ε=0.15, dans `tau-sweep` et dans `alpha-sweep` à α=5).

---

## 1. Sensibilité à ε (`epsilon-sweep`)

```
python param_sweep.py epsilon-sweep instances/instance_period2.json \
    runs/r2/baseline/period2_baseline_v2.meta.json \
    --epsilons 0.05,0.1,0.15,0.2,0.3,0.4
```

C* = 32.1881, indépendant de ε.

| ε | C* | threshold | corridor_size | winner_idx | winner_C_config | winner_Q | winner_Phi |
|---|---|---|---|---|---|---|---|
| 0.05 | 32.1881 | 33.7975 | 16 | 85 | 29.32 | 18.75 | 5.8730 |
| 0.1 | 32.1881 | 35.4069 | 109 | 180 | 31.32 | 17.03 | 6.2255 |
| 0.15 | 32.1881 | 37.0163 | 315 | 199 | 32.67 | 17.00 | 6.4571 |
| 0.2 | 32.1881 | 38.6257 | 660 | 2194 | 34.52 | 16.90 | 6.5525 |
| 0.3 | 32.1881 | 41.8445 | 1609 | 144 | 37.79 | 17.00 | 6.7659 |
| 0.4 | 32.1881 | 45.0633 | 2671 | 2045 | 40.16 | 17.63 | 6.8769 |

**Observation.** Le corridor croît de 16 configurations (0.4 % de l'univers)
à ε=0.05, à 2671 (67.0 %) à ε=0.4, une croissance encore plus rapide en
proportion que sur `period1` (qui partait de 46/3024 = 1.5 %). À ε=0.15
(valeur par défaut de l'instance), le corridor reste très sélectif :
315 configurations, soit 7.9 % de l'univers.

**Point notable : `winner_Q` n'est pas monotone en ε.** Il baisse d'abord
(18.75 → 17.03 → 17.00 → 16.90 entre ε=0.05 et ε=0.2), puis remonte
(17.00 à ε=0.3, 17.63 à ε=0.4). Un corridor plus large laisse à la Phase 2
plus de marge pour privilégier la flexibilité (Φ croît continûment de
5.8730 à 6.8769) au prix, passé un certain seuil, d'un léger retour en
arrière sur la performance brute, un comportement qui n'était pas visible
sur `period1` (où `winner_Q` restait globalement décroissant sur la
plage testée).

---

## 2. Sensibilité à τ (`tau-sweep`)

```
python param_sweep.py tau-sweep instances/instance_period2.json \
    runs/r2/baseline/period2_baseline_v2.meta.json \
    --taus 1,2,3,5,8,13,21,50,100
```

Corridor constant (315 configurations) sur tout le balayage.

### 2.1 Gagnant du corridor vs. τ

| τ | corridor_size | winner_idx | winner_C_config | winner_Q | winner_Theta | winner_Phi |
|---|---|---|---|---|---|---|
| 1 | 315 | 573 | 32.60 | 17.86 | 4.9308 | 4.4419 |
| 2 | 315 | 573 | 32.60 | 17.86 | 4.9308 | 4.4159 |
| 3 | 315 | 6 | 32.15 | 18.22 | 4.1364 | 4.6786 |
| 5 | 315 | 1715 | 32.74 | 17.50 | 4.1845 | 5.0995 |
| 8 | 315 | 1814 | 32.79 | 17.50 | 3.7800 | 6.1749 |
| 13 | 315 | 142 | 32.75 | 17.00 | 2.7300 | 6.7659 |
| 21 | 315 | 2042 | 32.60 | 17.70 | 1.8255 | 6.2913 |
| 50 | 315 | 47 | 30.75 | 17.00 | 1.7300 | 6.8136 |
| 100 | 315 | 47 | 30.75 | 17.00 | 1.7300 | 6.8136 |

**Différence marquante avec `period1` : le gagnant ne se stabilise qu'à
τ=50 ici**, pas à τ=13. Il change encore entre τ=21 (`2042`) et τ=50
(`47`), 7 gagnants distincts sur les 8 premières valeurs testées, contre
seulement 5 sur `period1` avant sa stabilisation à τ=13. Ce n'est qu'entre
τ=50 et τ=100 que le résultat est identique (`config_idx=47` aux deux
valeurs, Φ=6.8136 fixe).

### 2.2 Distribution de la taille de voisinage sur l'univers complet

| τ | min | p25 | médiane | moyenne | p75 | p90 | max | frac_isolés |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 2 | 1.95 | 2 | 3 | 3 | 3.0% |
| 2 | 2 | 5 | 6 | 6.23 | 8 | 9 | 10 | 0.0% |
| 3 | 5 | 10 | 13 | 13.19 | 16 | 18 | 22 | 0.0% |
| 5 | 9 | 19 | 25 | 24.74 | 31 | 35 | 43 | 0.0% |
| 8 | 14 | 31 | 38 | 35.47 | 41 | 47 | 52 | 0.0% |
| 13 | 15 | 38 | 39 | 45.37 | 58 | 70 | 78 | 0.0% |
| 21 | 28 | 63 | 79 | 82.58 | 99 | 119 | 128 | 0.0% |
| 50 | 31 | 79 | 79 | 98.35 | 127 | 159 | 159 | 0.0% |
| 100 | 31 | 79 | 79 | 98.35 | 127 | 159 | 159 | 0.0% |

Ici, la stabilisation du gagnant (τ=50) coïncide exactement avec la
saturation structurelle des voisinages (moyenne 98.35, p90/max=159/159,
identiques entre τ=50 et τ=100). C'est l'inverse du constat fait sur
`period1`, où le gagnant se stabilisait avant la saturation complète
(τ=13 contre saturation à τ≈50). Conclusion pratique importante : la
règle "τ∈[13,21] suffit" tirée de `period1` ne se généralise pas, sur
`period2` il faut aller jusqu'à la saturation structurelle complète (τ≥50)
pour un gagnant stable. Le choix de τ doit donc être validé par instance,
pas fixé une fois pour toutes.

---

## 3. Sensibilité à `phi_top_fraction` (`phi-fraction-sweep`, simulation)

```
python param_sweep.py phi-fraction-sweep instances/instance_period2.json \
    runs/r2/baseline/period2_baseline_v2.meta.json \
    --fractions 0.1,0.2,0.3,0.5,0.7,1.0
```

Gagnant de référence (fraction=100 %) : **`config_idx=199`**.

| fraction | kept_n | C* (simulé) | corridor_size | winner_idx | winner==baseline | écart de coût vs vrai C* |
|---|---|---|---|---|---|---|
| 10% | 399 | 37.5067 | 95 | 2327 | False | +32.42% |
| 20% | 798 | 35.4267 | 119 | 2232 | False | +26.21% |
| 30% | 1197 | 34.5467 | 139 | 143 | False | +22.69% |
| 50% | 1995 | 33.0081 | 95 | 1757 | False | +16.52% |
| 70% | 2793 | 32.4881 | 164 | 199 | **True** | +14.07%¹ |
| 100% | 3990 | 32.1881 | 315 | 199 | True | +14.07%¹ |

¹ Écart normal Phase 1/Phase 2 (le gagnant Phase 2 n'est pas le
minimiseur de coût pur), pas une erreur d'approximation, identique à
70 % et 100 % puisque le winner retrouvé est le même.

**Amélioration notable par rapport à `period1` : le vrai gagnant est
retrouvé dès 70 %** (contre jamais, même à 70 %, sur `period1`). En
dessous de 70 %, l'écart de coût est franchement dégradé (+16.5 % à 50 %,
jusqu'à +32.4 % à 10 %), le défaut `--phi-top-fraction 0.3`
d'`orchestrate.py` reste risqué ici aussi (+22.7 %).

---

## 4. Recall structurel du corridor (`heuristic_analysis.py phi-recall`)

```
python heuristic_analysis.py phi-recall instances/instance_period2.json \
    runs/r2/baseline/period2_baseline_v2.meta.json \
    --fractions 0.1,0.2,0.3,0.5,0.7,1.0
```

Baseline : C* = 32.1881, corridor size = 315, winner config_idx = 199.

| fraction | kept_n | corridor_recall | winner_included |
|---|---|---|---|
| 10% | 399 | 0.0% | False |
| 20% | 798 | 2.5% | True |
| 30% | 1197 | 10.5% | True |
| 50% | 1995 | 19.7% | True |
| 70% | 2793 | 41.3% | True |
| 100% | 3990 | 100.0% | True |

Le gagnant `199` est présent dans le sous-univers dès 20 %, mais — comme
sur `period1`, le corridor recalculé sur ce sous-univers tronqué reste
très incomplet jusqu'à 70 % (41.3 % de recall), ce qui explique pourquoi
la section 3 ne retrouve le bon arbitrage qu'à 70 %, pas à 20 %.

---

## 5. Sensibilité à α — Phase 1 vs. Phase 2 (`alpha-sweep`)

```
python heuristic_analysis.py alpha-sweep instances/instance_period2.json \
    runs/r2/baseline/period2_baseline_v2.meta.json \
    --alphas 1,2,...,20 --top-k 20
```

Grille fine d'emblée (pas de 1, α=1 à 20), reference_alpha=5.

### 5.1 Table complète

| α | C* | corridor_size | winner_idx (P2) | winner_C_config | winner_Q | phase1_winner_idx | phase1_C_config | phase1_Q | P1==P2 | topK_jaccard | topK_spearman | topK_kendall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 28.7176 | 219 | 1719 | 31.26 | 17.50 | 0 | 27.85 | 18.22 | **False** | 0.905 | 0.988 | 0.947 |
| 2 | 29.5852 | 239 | 1625 | 32.25 | 17.50 | 0 | 27.85 | 18.22 | **False** | 0.905 | 0.992 | 0.958 |
| 3 | 30.4529 | 259 | 1625 | 32.25 | 17.50 | 0 | 27.85 | 18.22 | **False** | 0.905 | 0.992 | 0.958 |
| 4 | 31.3205 | 288 | 199 | 32.67 | 17.00 | 0 | 27.85 | 18.22 | **False** | 0.905 | 0.997 | 0.979 |
| 5 | 32.1881 | 315 | 199 | 32.67 | 17.00 | 0 | 27.85 | 18.22 | **False** | 1.000 | 1.000 | 1.000 |
| 6 | 33.0557 | 340 | 199 | 32.67 | 17.00 | 0 | 27.85 | 18.22 | **False** | 0.905 | 0.994 | 0.958 |
| 7 | 33.9233 | 367 | 199 | 32.67 | 17.00 | 0 | 27.85 | 18.22 | **False** | 0.739 | 0.970 | 0.879 |
| 8 | 34.7910 | 395 | 1757 | 33.34 | 17.50 | 0 | 27.85 | 18.22 | **False** | 0.667 | 0.926 | 0.779 |
| 9 | 35.6586 | 420 | 1757 | 33.34 | 17.50 | 0 | 27.85 | 18.22 | **False** | 0.667 | 0.911 | 0.747 |
| 10 | 36.5262 | 448 | 1757 | 33.34 | 17.50 | 0 | 27.85 | 18.22 | **False** | 0.600 | 0.901 | 0.716 |
| 11 | 37.3938 | 482 | 1757 | 33.34 | 17.50 | 0 | 27.85 | 18.22 | **False** | 0.600 | 0.892 | 0.695 |
| 12 | 38.2614 | 514 | 1757 | 33.34 | 17.50 | 0 | 27.85 | 18.22 | **False** | 0.538 | 0.877 | 0.684 |
| 13 | 39.1290 | 549 | 2194 | 34.52 | 16.90 | 0 | 27.85 | 18.22 | **False** | 0.538 | 0.872 | 0.674 |
| 14 | 39.9967 | 591 | 2194 | 34.52 | 16.90 | 0 | 27.85 | 18.22 | **False** | 0.538 | 0.859 | 0.663 |
| 15 | 40.8643 | 641 | 237 | 34.75 | 17.00 | 0 | 27.85 | 18.22 | **False** | 0.538 | 0.848 | 0.653 |
| 16 | 41.7319 | 674 | 237 | 34.75 | 17.00 | 0 | 27.85 | 18.22 | **False** | 0.538 | 0.836 | 0.642 |
| 17 | 42.5995 | 714 | 237 | 34.75 | 17.00 | 0 | 27.85 | 18.22 | **False** | 0.538 | 0.823 | 0.632 |
| 18 | 43.4586 | 756 | 237 | 34.75 | 17.00 | 14 | 28.03 | 18.00 | **False** | 0.481 | 0.809 | 0.611 |
| 19 | 44.3157 | 791 | 237 | 34.75 | 17.00 | 14 | 28.03 | 18.00 | **False** | 0.481 | 0.809 | 0.611 |
| 20 | 45.1729 | 843 | 237 | 34.75 | 17.00 | 14 | 28.03 | 18.00 | **False** | 0.481 | 0.809 | 0.611 |

**`P1==P2` reste `False` sur les 20 valeurs testées** — même constat
systématique que sur `period1`.

### 5.2 Trajectoire du gagnant Phase 1 (argmin coût pur)

Seulement **2 gagnants distincts** sur toute la plage α∈[1,20]: bien
plus stable que `period1` (3 gagnants, fenêtres étroites de 2 unités) :

| Plage d'α | phase1_winner_idx | C_config | Q |
|---|---|---|---|
| [1, 17] | 0 | 27.85 | 18.22 |
| [18, 20] | 14 | 28.03 | 18.00 |

Bascule unique entre α=17 et α=18.

### 5.3 Intervalles de stabilité du gagnant Phase 2

| winner_idx | plage d'α | largeur |
|---|---|---|
| 1719 | [1, 1] | 1 |
| 1625 | [2, 3] | 2 |
| 199 | [4, 7] | 4 |
| 1757 | [8, 12] | 5 |
| 2194 | [13, 14] | 2 |
| 237 | [15, 20] | 6 |

**Nettement plus stable que `period1`**, où aucun intervalle ne dépassait
2 unités : ici deux paliers dépassent 4 unités (`199` sur [4,7] et
`1757` sur [8,12]), et `237` reste gagnant sur toute la moitié haute de
la plage testée ([15,20], largeur 6). Le gagnant à l'α de référence
(α=5) est `199`, stable sur [4,7], un plateau confortable comparé au
[5,6] fragile trouvé sur `period1`.

La dégradation de `topK_jaccard`/`spearman`/`kendall` en s'éloignant de
α=5 suit le même profil monotone que sur `period1`, mais reste
globalement plus élevée à distance égale (ex. jaccard=0.481 à α=20 ici
contre 0.212 sur `period1`), cohérent avec un classement plus stable
sur cette instance.

---

## 6. Visualisation Pareto (`pareto2.png`)

```
python heuristic_analysis.py pareto instances/instance_period2.json \
    runs/r2/baseline/period2_baseline_v2.meta.json \
    --alphas 1,2,...,20 --output-png runs/r2/baseline/pareto2.png
```

Front de Pareto (**14 configurations non dominées sur (C_config, Q)**,
soit 0.4 % de l'univers, un front deux fois plus large que sur `period1`,
7 configs) :

| config_idx | C_config | Q |
|---|---|---|
| 0 | 27.85 | 18.22 |
| 14 | 28.03 | 18.00 |
| 489 | 28.83 | 17.86 |
| 1615 | 28.91 | 17.50 |
| 1619 | 29.21 | 17.50 |
| 19 | 29.85 | 17.00 |
| 23 | 30.15 | 17.00 |
| 2090 | 31.70 | 16.90 |
| 2167 | 33.20 | 16.90 |
| 2109 | 33.70 | 16.90 |
| 2204 | 35.70 | 16.90 |
| 2356 | 36.35 | 16.90 |
| 2375 | 37.70 | 16.90 |
| 2403 | 40.52 | 16.90 |

La grille α utilisée pour ce plot (1 à 20, pas de 1) couvre cette fois
l'intégralité des paliers identifiés en section 5 : les **2** gagnants
Phase 1 distincts (`0`, `14`) et les **6** gagnants Phase 2 distincts
(`1719`, `1625`, `199`, `1757`, `2194`, `237`) devraient donc tous
apparaître comme marqueurs sur le graphe, contrairement au premier
essai sur `period1` où une grille trop grossière avait fait sauter le
gagnant `#0`.

---

## 7. Étude qualitative (à faire)

Pas encore lancé pour `period2`. Deux paires ressortent naturellement de
la section 5 :

```bash
# Phase 2 vs Phase 1 à l'alpha de référence (5)
python config_diff.py instances/instance_period2.json \
    --run runs/r2/baseline/period2_baseline_v2.meta.json --ids 199,0

# les deux gagnants du plateau haut-alpha ([15,20]) : Phase 2 (237) vs
# Phase 1 (14), tous deux stables sur une large plage d'alpha
python config_diff.py instances/instance_period2.json \
    --run runs/r2/baseline/period2_baseline_v2.meta.json --ids 237,14
```

---

## 8. Synthèse

| Paramètre | Plage stable / recommandée sur `period2` | Comparaison à `period1` |
|---|---|---|
| ε | ≤0.15 pour rester sélectif (7.9 % de l'univers) | Croissance du corridor encore plus rapide en proportion |
| τ | ≥50 (stabilisation coïncide avec la saturation structurelle) | **Contraire à `period1`** (stable dès τ=13, avant saturation à ~50) |
| phi_top_fraction | ≥70 % | Légèrement meilleur que `period1` (jamais retrouvé même à 70 %) |
| α | Plateaux larges (jusqu'à 6 unités pour `237`) | Nettement plus stable que `period1` (jamais >2 unités) |

**Constat transversal, confirmé sur les deux instances :** `P1==P2` est
`False` sans exception sur toute la plage α testée, la Phase 2 modifie
systématiquement le choix.

**Constat le plus important de cette deuxième instance : aucun des seuils
"sûrs" trouvés sur `period1` ne se généralise tel quel.** Le τ qui
suffisait sur `period1` (13) est très insuffisant ici (il faut 50), alors
que d'autres aspects (stabilité du gagnant en α, recouvrement de
`phi_top_fraction`) sont au contraire meilleurs ici. C'est cohérent
avec la mise en garde du README (section 11) sur l'instance-dépendance
de `partial-inversion`, mais on voit ici que cette dépendance s'étend
aussi à τ, pas seulement à `phi_top_fraction`. Un τ "universel" choisi
sur une seule instance de calibration serait risqué : il faudrait au
minimum reproduire le `tau-sweep` sur chaque nouvelle instance/période
avant de figer sa valeur pour l'étude finale.

---

