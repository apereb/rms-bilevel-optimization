# Analyse de sensibilité — RMS bi-level, `instance_period1`

Run de référence : `runs/r1/baseline/period1_baseline_v2.meta.json` (mode `full`,
aucun appel Gurobi supplémentaire dans cette analyse tout ce qui suit vient
de `recompute`-style sweeps sur ce baseline).

**Univers de configurations feasibles : 3024**

Sauf mention contraire, les paramètres non balayés sont ceux du fichier
d'instance : α, ε et τ à leur valeur enregistrée.

---

## 1. Sensibilité à ε (`epsilon-sweep`)

```
python param_sweep.py epsilon-sweep instances/instance_period1.json \
    runs/r1/baseline/period1_baseline_v2.meta.json \
    --epsilons 0.05,0.1,0.15,0.2,0.3,0.4
```

C* est indépendant de ε par construction (23.9781 sur toute la plage), seul
le corridor et le gagnant Phase 2 bougent.

| ε | C* | threshold | corridor_size | winner_idx | winner_C_config | winner_Q | winner_Phi |
|---|---|---|---|---|---|---|---|
| 0.05 | 23.9781 | 25.1770 | 46 | 27 | 19.09 | 24.50 | 2.3945 |
| 0.1 | 23.9781 | 26.3759 | 199 | 108 | 20.54 | 24.50 | 2.9795 |
| 0.15 | 23.9781 | 27.5748 | 526 | 1419 | 21.49 | 25.00 | 3.1409 |
| 0.2 | 23.9781 | 28.7737 | 986 | 231 | 22.90 | 24.50 | 3.8659 |
| 0.3 | 23.9781 | 31.1715 | 2041 | 1797 | 25.57 | 22.63 | 4.1687 |
| 0.4 | 23.9781 | 33.5693 | 2746 | 1833 | 26.16 | 22.63 | 4.2051 |

**Observation.** Le corridor croît très vite avec ε : de 46 configurations
(1.5 % de l'univers) à ε=0.05, à 2746 (90.8 % de l'univers) à ε=0.4. Le
comportement n'est pas linéaire: entre ε=0.15 et ε=0.3, le corridor passe
de 526 à 2041 configurations pour un déplacement du seuil de seulement
27.57 → 31.17 (+13 %). Ceci indique une masse importante de configurations
dont le coût total est resserré autour de C*, dans une bande relativement
étroite.

**Implication pratique.** Au-delà de ε≈0.2, le corridor perd son
interprétation de "petit ensemble de candidats proches du meilleur" : à
ε=0.4 il représente déjà l'essentiel de l'univers. Les valeurs ε∈[0.05, 0.15]
sont les seules où le corridor reste un filtre sélectif sur cette instance ;
ε=0.15 (valeur testée qui correspond aussi à l'instance de référence pour le
reste des sweeps ci-dessous, winner_idx=1419) semble un compromis raisonnable
entre sélectivité et marge de tolérance.

---

## 2. Sensibilité à τ (`tau-sweep`)

```
python param_sweep.py tau-sweep instances/instance_period1.json \
    runs/r1/baseline/period1_baseline_v2.meta.json \
    --taus 1,2,3,5,8,13,21,50,100
```

α et ε restent fixes ⇒ le corridor lui-même est constant (526
configurations) sur tout le balayage ; seul Φ (et donc le gagnant Phase 2
au sein de ce corridor) varie avec τ.

### 2.1 Gagnant du corridor vs. τ

| τ | corridor_size | winner_idx | winner_C_config | winner_Q | winner_Theta | winner_Phi |
|---|---|---|---|---|---|---|
| 1 | 526 | 1404 | 21.13 | 25.00 | 3.6659 | 3.0659 |
| 2 | 526 | 147 | 21.57 | 24.50 | 3.1159 | 3.1159 |
| 3 | 526 | 1419 | 21.49 | 25.00 | 3.6159 | 3.1409 |
| 5 | 526 | 1620 | 21.65 | 23.80 | 3.1087 | 3.0629 |
| 8 | 526 | 96 | 20.98 | 24.50 | 1.7758 | 3.2633 |
| 13 | 526 | 81 | 20.65 | 24.50 | 1.7258 | 3.1012 |
| 21 | 526 | 81 | 20.65 | 24.50 | 1.7258 | 3.3473 |
| 50 | 526 | 81 | 20.65 | 24.50 | 1.7258 | 3.3473 |
| 100 | 526 | 81 | 20.65 | 24.50 | 1.7258 | 3.3473 |

Le gagnant se stabilise à `config_idx=81` dès τ=13 et ne bouge plus
jusqu'à τ=100 (Φ lui-même se stabilise à 3.3473 dès τ=21). En dessous de
τ=13, le gagnant est instable : 5 valeurs différentes (1404, 147, 1419,
1620, 96) pour les 5 premiers τ testés (1, 2, 3, 5, 8).

### 2.2 Distribution de la taille de voisinage sur l'univers complet

| τ | min | p25 | médiane | moyenne | p75 | p90 | max | frac_isolés |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 1 | 1.31 | 2 | 2 | 2 | 11.9% |
| 2 | 2 | 3 | 4 | 3.98 | 5 | 5 | 6 | 0.0% |
| 3 | 4 | 7 | 7 | 7.81 | 9 | 11 | 14 | 0.0% |
| 5 | 6 | 13 | 17 | 17.51 | 22 | 26 | 33 | 0.0% |
| 8 | 10 | 24 | 33 | 34.05 | 43 | 52 | 69 | 0.0% |
| 13 | 17 | 44 | 57 | 61.87 | 83 | 98 | 119 | 0.0% |
| 21 | 23 | 59 | 87 | 94.08 | 139 | 170 | 179 | 0.0% |
| 50 | 23 | 59 | 89 | 95.67 | 143 | 179 | 179 | 0.0% |
| 100 | 23 | 59 | 89 | 95.67 | 143 | 179 | 179 | 0.0% |

**Saturation structurelle atteinte entre τ=21 et τ=50** : la moyenne passe
de 94.08 à 95.67 puis reste exactement identique (95.67, max=179) entre
τ=50 et τ=100, plus aucune paire de configurations ne devient
mutuellement atteignable au-delà de τ=50. `frac_isolés` tombe à 0 % dès
τ=2 : au-delà de τ=1, plus aucune configuration n'est totalement isolée.

**Point clé.** La sélection (le gagnant) se stabilise à τ=13, soit
bien avant la saturation structurelle complète des voisinages (τ≈50).
Il n'est donc pas nécessaire d'attendre τ→∞ (ou même le plateau à τ=50)
pour obtenir un résultat de sélection stable : τ∈[13, 21] suffit en
pratique sur cette instance.

---

## 3. Sensibilité à `phi_top_fraction` (`phi-fraction-sweep`, simulation sans Gurobi)

```
python param_sweep.py phi-fraction-sweep instances/instance_period1.json \
    runs/r1/baseline/period1_baseline_v2.meta.json \
    --fractions 0.1,0.2,0.3,0.5,0.7,1.0
```

Gagnant de référence (fraction=100%, = le vrai gagnant du baseline) :
**`config_idx=1419`**.

| fraction | kept_n | C* (simulé) | corridor_size | winner_idx | winner==baseline | écart de coût vs vrai C* |
|---|---|---|---|---|---|---|
| 10% | 302 | 28.3733 | 207 | 1833 | False | +31.57% |
| 20% | 605 | 26.7333 | 245 | 285 | False | +26.34% |
| 30% | 907 | 26.3733 | 276 | 285 | False | +26.34% |
| 50% | 1512 | 26.1833 | 500 | 1743 | False | +23.77% |
| 70% | 2117 | 24.3733 | 229 | 162 | False | +16.50% |
| 100% | 3024 | 23.9781 | 526 | 1419 | True | +14.45%¹ |

¹ L'écart de +14.45% à 100% n'est pas une erreur : `total_cost` au gagnant
Phase 2 (`1419`) est par construction ≥ C* (Phase 2 sélectionne par argmax Φ
dans le corridor, pas par coût minimal) c'est l'écart normal entre le
gagnant Phase 1 pur et le gagnant Phase 2 réel, cohérent avec la section 5.

**Observation critique.** Le vrai gagnant (`1419`) n'est retrouvé à
aucune fraction inférieure à 100 % parmi celles testées. Même à 70 % de
l'univers gardé, l'écart de coût reste de +16.5 %. Le défaut de
`orchestrate.py` pour `partial-inversion` (`--phi-top-fraction 0.3`)
donnerait ici un gagnant à +26.3 % du vrai C* une dégradation significative,
pas seulement un delta marginal.

---

## 4. Recall structurel du corridor (`heuristic_analysis.py phi-recall`)

```
python heuristic_analysis.py phi-recall instances/instance_period1.json \
    runs/r1/baseline/period1_baseline_v2.meta.json \
    --fractions 0.1,0.2,0.3,0.5,0.7,1.0
```

Baseline : C* = 23.9781, corridor size = 526, winner config_idx = 1419.

| fraction | kept_n | corridor_recall | winner_included |
|---|---|---|---|
| 10% | 302 | 0.0% | False |
| 20% | 605 | 2.1% | True |
| 30% | 907 | 4.6% | True |
| 50% | 1512 | 9.7% | True |
| 70% | 2117 | 28.9% | True |
| 100% | 3024 | 100.0% | True |

**Lecture croisée avec la section 3.** Ce tableau dit que le *config_idx*
1419 (le vrai gagnant) fait partie des configurations gardées dès 20 %
`winner_included=True`. Pourtant, à la section 3, le gagnant simulé à 20 %
est `285`, pas `1419` : le vrai gagnant survit au filtre Φ, mais un autre
candidat, absent du corridor réel à 100 % ou de coût plus favorable dans le
sous-univers tronqué, prend sa place. Le problème n'est donc pas que
`1419` soit éliminé par le filtre Φ, il ne l'est jamais après 10 %, mais
que le corridor recalculé sur un sous-univers tronqué (`corridor_recall`
seulement 2.1 % à 9.7 % pour 20–50 %) est trop incomplet pour que la
comparaison de coût aboutisse au bon arbitrage. Le vrai corridor complet
(526 configs) n'est recouvré qu'à moins de 30 % même à 70 % de l'univers
gardé (28.9 % de recall).

**Conclusion.** `partial-inversion` n'est pas fiable sur `instance_period1`
en-dessous d'environ 70–100 % de l'univers, à ce stade le gain de temps de
calcul devient marginal. Le filtre Φ seul n'est donc utilisable qu'au coût d'une perte certaine de la qualité de la solution.

---

## 5. Sensibilité à α — Phase 1 vs. Phase 2 (`alpha-sweep`)

Deux balayages ont été effectués, tous deux avec `--top-k 20` et
`reference_alpha=5` (valeur par défaut = α de l'instance) :

```
python heuristic_analysis.py alpha-sweep instances/instance_period1.json \
    runs/r1/baseline/period1_baseline_v2.meta.json \
    --alphas 1,2,3,4,5,6,7,8,10,15,20 --top-k 20

python heuristic_analysis.py alpha-sweep instances/instance_period1.json \
    runs/r1/baseline/period1_baseline_v2.meta.json \
    --alphas 10,11,12,13,14,15,16,17,18,19,20 --top-k 20
```

### 5.1 Table fusionnée (grille grossière + grille fine 10–20)

| α | C* | corridor_size | winner_idx (P2) | winner_C_config | winner_Q | phase1_winner_idx | phase1_C_config | phase1_Q | P1==P2 | topK_jaccard | topK_spearman | topK_kendall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 19.1476 | 286 | 108 | 20.54 | 24.50 | 324 | 17.94 | 25.36 | **False** | 0.739 | 0.947 | 0.842 |
| 2 | 20.3552 | 328 | 120 | 21.02 | 24.50 | 324 | 17.94 | 25.36 | **False** | 0.739 | 0.961 | 0.863 |
| 3 | 21.5629 | 391 | 135 | 21.09 | 24.50 | 324 | 17.94 | 25.36 | **False** | 0.818 | 0.980 | 0.905 |
| 4 | 22.7705 | 460 | 159 | 21.49 | 24.50 | 324 | 17.94 | 25.36 | **False** | 0.905 | 0.988 | 0.937 |
| 5 | 23.9781 | 526 | 1419 | 21.49 | 25.00 | 324 | 17.94 | 25.36 | **False** | 1.000 | 1.000 | 1.000 |
| 6 | 25.1857 | 595 | 1419 | 21.49 | 25.00 | 324 | 17.94 | 25.36 | **False** | 1.000 | 0.973 | 0.916 |
| 7 | 26.3933 | 668 | 162 | 22.10 | 24.50 | 324 | 17.94 | 25.36 | **False** | 0.905 | 0.910 | 0.800 |
| 8 | 27.6010 | 744 | 162 | 22.10 | 24.50 | 324 | 17.94 | 25.36 | **False** | 0.905 | 0.860 | 0.737 |
| 10 | 30.0162 | 902 | 216 | 22.54 | 24.50 | 324 | 17.94 | 25.36 | **False** | 0.667 | 0.750 | 0.632 |
| 11 | 31.2238 | 1004 | 231 | 22.90 | 24.50 | 324 | 17.94 | 25.36 | **False** | 0.600 | 0.633 | 0.568 |
| 12 | 32.4314 | 1111 | 231 | 22.90 | 24.50 | 324 | 17.94 | 25.36 | **False** | 0.538 | 0.574 | 0.516 |
| 13 | 33.6390 | 1198 | 267 | 23.49 | 24.50 | 324 | 17.94 | 25.36 | **False** | 0.538 | 0.574 | 0.516 |
| 14 | 34.8467 | 1295 | 1743 | 24.01 | 23.80 | 324 | 17.94 | 25.36 | **False** | 0.481 | 0.537 | 0.474 |
| 15 | 36.0400 | 1379 | 1743 | 24.01 | 23.80 | 0 | 18.54 | 24.50 | **False** | 0.429 | 0.442 | 0.411 |
| 16 | 37.2067 | 1454 | 1779 | 24.60 | 23.80 | 0 | 18.54 | 24.50 | **False** | 0.429 | 0.353 | 0.368 |
| 17 | 38.3490 | 1528 | 1797 | 25.57 | 22.63 | 1836 | 19.05 | 23.84 | **False** | 0.333 | 0.317 | 0.347 |
| 18 | 39.4843 | 1585 | 1797 | 25.57 | 22.63 | 1836 | 19.05 | 23.84 | **False** | 0.250 | 0.290 | 0.326 |
| 19 | 40.6195 | 1650 | 1833 | 26.16 | 22.63 | 1836 | 19.05 | 23.84 | **False** | 0.250 | 0.283 | 0.316 |
| 20 | 41.7548 | 1711 | 1833 | 26.16 | 22.63 | 1836 | 19.05 | 23.84 | **False** | 0.212 | 0.275 | 0.305 |

**`P1==P2` est `False` sur les 19 valeurs d'α testées, sans exception.** Le
critère de flexibilité (Φ) ne départage jamais un cas limite sur cette
instance : il change systématiquement le choix par rapport au pur
minimiseur de coût, quel que soit α entre 1 et 20.

### 5.2 Trajectoire du gagnant Phase 1 (argmin coût pur)

Trois plateaux identifiés sur α∈[1, 20] :

| Plage d'α | phase1_winner_idx | C_config | Q |
|---|---|---|---|
| [1, 14] | 324 | 17.94 | 25.36 |
| [15, 16] | 0 | 18.54 | 24.50 |
| [17, 20] | 1836 | 19.05 | 23.84 |

La bascule 324 → 0 se produit entre α=14 et α=15 ; la bascule 0 → 1836 se
produit entre α=16 et α=17. Ces deux fenêtres sont étroites (2 valeurs
d'α chacune), c'est pour cela que, sur la grille `1,3,5,8,11,14,17,20`
utilisée pour le plot Pareto (section 6), le gagnant `#0` n'apparaît à
aucune des valeurs testées : la grille saute directement de α=14 (encore
`#324`) à α=17 (déjà `#1836`).

### 5.3 Intervalles de stabilité du gagnant Phase 2

D'après la grille grossière (1–20, pas variable) :

| winner_idx | plage d'α |
|---|---|
| 108 | [1, 1] |
| 120 | [2, 2] |
| 135 | [3, 3] |
| 159 | [4, 4] |
| **1419** | **[5, 6]** |
| 162 | [7, 8] |
| 216 | [10, 10] |
| 1743 | [15, 15] |
| 1833 | [20, 20] |

D'après la grille fine (10–20, pas de 1) :

| winner_idx | plage d'α |
|---|---|
| 216 | [10, 10] |
| 231 | [11, 12] |
| 267 | [13, 13] |
| 1743 | [14, 15] |
| 1779 | [16, 16] |
| 1797 | [17, 18] |
| 1833 | [19, 20] |

Le gagnant α=5 (l'α enregistré dans l'instance) est `1419`, stable sur
[5, 6] seulement un intervalle de largeur 1, le plus étroit avec celui
de winner=1743. Sur toute la plage testée (1 à 20), aucun intervalle de
stabilité ne dépasse 2 unités d'α une fois qu'on affine la grille (la
grille grossière sous-estimait la fragmentation : elle montrait par exemple
un unique palier [15,15] pour 1743 là où la grille fine révèle [14,15]).
Le choix du gagnant est donc fragile vis-à-vis d'α sur cette instance
aucune valeur ne s'impose comme un plateau large et robuste.

La cohérence du classement (topK_jaccard/spearman/kendall vs. la référence
α=5) se dégrade progressivement et de façon monotone à mesure qu'on s'éloigne
de α=5 dans les deux directions : jaccard tombe de 1.000 (α=5,6) à 0.212
à α=20, et à 0.739 dès α=1-2. Le classement des 20 meilleures configurations
n'est donc représentatif que dans un voisinage restreint de α=5.

---

## 6. Visualisation Pareto (`pareto2.png`)

```
python heuristic_analysis.py pareto instances/instance_period1.json \
    runs/r1/baseline/period1_baseline_v2.meta.json \
    --alphas 1,3,5,8,11,14,17,20 --output-png runs/r1/baseline/pareto2.png
```

Front de Pareto (7 configurations non dominées sur (C_config, Q), soit
0.2 % de l'univers) :

| config_idx | C_config | Q |
|---|---|---|
| 324 | 17.94 | 25.36 |
| 0 | 18.54 | 24.50 |
| 1836 | 19.05 | 23.84 |
| 1512 | 19.65 | 23.80 |
| 972 | 20.58 | 23.22 |
| 1998 | 22.01 | 23.09 |
| 1782 | 25.21 | 22.63 |

Sur le graphe, les gagnants Phase 1 (losanges orange) tombent bien sur le
front, comme attendu théoriquement pour une scalarisation linéaire de
(C_config, Q). Seuls 2 losanges distincts apparaissent (`#324` à α=1, et
`#1836` à α=17) cohérent avec la section 5.2 : la grille testée
(1,3,5,8,11,14,17,20) ne traverse jamais la fenêtre étroite [15,16] où
`#0` est optimal. Les gagnants Phase 2 (points noirs), en revanche,
apparaissent bien à chacune des 8 valeurs d'α, puisqu'ils changent à
(presque) chaque pas sur cette grille (`108/135` pour α=1/3, `1419` pour
α=5, `162` pour α=8, `231` pour α=11, `1743` pour α=14, `1797` pour α=17,
`1833` pour α=20).

L'écart visuel entre un point noir et le losange orange le plus proche à un
même α est exactement le "prix" payé par la Phase 2 en coût/makespan pour
gagner en flexibilité (Φ) à cet α.

---

## 7. Étude qualitative — diff `#1797` vs. `#1836`

```
python config_diff.py instances/instance_period1.json \
    --run runs/r1/baseline/period1_baseline_v2.meta.json --ids 1797,1836
```

Ces deux configurations sont proches en coût (25.57 vs 19.05, gagnantes
Phase 2/Phase 1 respectivement autour de α=17) mais non mutuellement
atteignables par reconfiguration : `T_rec` = +∞ dans les deux sens.

| stage | machine #1797 → #1836 | config #1797 → #1836 | n #1797 → #1836 | modules ajoutés | modules retirés | modules partagés |
|---|---|---|---|---|---|---|
| 0 | 2→2 (identique) | 4→4 | 3→3 | — | — | [10,14,15,18,23] |
| 1 | 4→4 (identique) | 2→2 | 3→3 | — | — | [10,15,16,20,24] |
| 2 | **2→3** | 3→1 | 3→1 | [10,13,16] | [23] | [15,19] |
| 3 | **3→2** | 2→3 | 3→3 | [23] | [11,12,16,21] | [15,19] |
| 4 | **2→1** | 2→1 | 1→1 | [12,17,20,22] | [11,15,19] | — |

Sur les 5 étages, 3 (étages 2, 3 et 4) changent de type de machine
(`machine_id`) c'est un changement structurel, pas une simple
retooling/re-jauge d'unités. C'est exactement ce qui rend `T_rec=+∞` :
d'après le modèle (section 4.3 du README), un changement de type de
machine à n'importe quel étage rend la paire inatteignable, quel que soit
τ. C_config baisse de 25.57 à 19.05 (Δ=-6.52) malgré ce changement radical
de structure un bon exemple concret pour illustrer, dans le mémoire,
pourquoi Φ pénalise l'isolement structurel même quand deux configurations
sont proches en coût.

---

## 8. Synthèse

| Paramètre | Plage stable / recommandée | Justification |
|---|---|---|
| ε | 0.05–0.15 | Au-delà, le corridor perd son caractère sélectif (>1500 configs dès ε=0.2) |
| τ | 13–21 | Le gagnant se stabilise à τ=13 ; la saturation structurelle complète (moyenne de voisinage) n'arrive qu'à τ≈50, mais n'est pas nécessaire pour un résultat de sélection stable. Attention, nous parlons bien ici de **plage de stabilité**. Dans un cas industriel, le choix de l'effort allouer à la reconfiguration pourrait être guider différement. |
| phi_top_fraction | ≥70–100 % pour rester fiable | Le vrai gagnant n'est jamais retrouvé en-dessous de 100 % sur les fractions testées ; écart de coût encore +16.5 % à 70 % |
| α | Fragile partout sur [1,20] | Aucun intervalle de stabilité du gagnant Phase 2 n'excède 2 unités d'α une fois la grille affinée |

---
