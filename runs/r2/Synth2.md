# Synthèse — Instance 2 (period2) : baseline vs. heuristiques

## Configuration expérimentale

- **Instance** : `instance_period2.json`
- **Baseline** (`full`) : 3990 configurations, `time_limit=15s`
- **Two-pass** : passe 1 à 5s (toutes configs), passe 2 à 15s sur le sous-ensemble tel que `total_cost ≤ (1+1.5ε)×best_provisoire`
- **Partial-inversion** : conservation des 30% de configurations au Φ le plus élevé avant scheduling (1197 configs)

## Résultats globaux

| Run | Configs planifiées | C* | Corridor | Gagnant | Coût du gagnant | Φ du gagnant |
|---|---|---|---|---|---|---|
| baseline | 3990 | 32.1881 | 315 | 199 | 36.7176 | 6.4571 |
| two-pass | 3990 | 32.2738 | 333 | 199 | 36.7176 | 6.4571 |
| partial-inversion | 1197 | 33.0081 | 75 | 1757 | 37.5067 | 6.5073 |

## Comparaisons deux-à-deux

| Paire | Jaccard (corridor) | Spearman | Kendall | Le 1er gagnant survit dans le corridor du 2nd | Écart de coût du gagnant du 2nd vs C* du 1er |
|---|---|---|---|---|---|
| baseline vs. two-pass | 0.946 | 1.000 | 0.992 | Oui | +14.07% |
| baseline vs. partial-inversion | 0.127 | 1.000 | 0.999 | Oui | +16.52% |
| partial-inversion vs. two-pass | 0.124 | 1.000 | 0.992 | Non | +11.24% |

## Interprétation

### Two-pass

`two-pass` retrouve exactement le même gagnant que la baseline (`199`) et le même coût total, avec un corridor très proche (Jaccard=0.946). Comme sur les autres instances testées, une passe grossière à 5s suivie d'un raffinement ciblé à 15s reproduit la qualité d'un run intégralement résolu à 15s, à moindre coût de calcul.

### Partial-inversion

Contrairement à ce qu'on observe sur d'autres instances, le gagnant de la baseline (`199`) fait bien partie du corridor retrouvé par `partial-inversion`, l'heuristique n'a pas ici conduit à une divergence de décision. Elle reste néanmoins coûteuse en couverture : seuls 12.7% du corridor réel sont retrouvés (Jaccard=0.127), et le gagnant propre à `partial-inversion` (`1757`) a un coût supérieur de 16.5% au véritable optimum.

### Classement

Spearman ≥ 1.000 et Kendall ≥ 0.992 sur les trois paires comparées : le classement relatif des configurations communément évaluées reste quasiment parfait, malgré des recouvrements de corridor très variables selon la paire. La dégradation observée pour `partial-inversion` est une question de couverture, pas de fiabilité du classement.

### Sensibilité à α

| α | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gagnant | 1719 | 1625 | 1625 | 199 | 199 | 199 | 199 | 1757 | 1757 | 237 | 237 |

Le gagnant réel (α=5) reste stable sur un intervalle α∈[4,7], plus large que sur d'autres instances testées, mais la configuration change dès qu'on s'en écarte significativement: la robustesse au choix d'α reste limitée.

### Pareto-optimalité

Sur les 3990 configurations évaluées, seules 14 (0.35%) sont Pareto-optimales sur le plan (C_config, Q). Le gagnant retenu n'en fait pas partie : il est dominé sur le plan coût/débit pur par d'autres configurations. Ceci est cohérent avec la logique de sélection en deux phase: le gagnant est choisi par argmax Φ à l'intérieur du corridor de coût, non par optimalité conjointe (coût, débit).

## Comparaison inter-instances

| | Instance 1 | Instance 2 |
|---|---|---|
| Intervalle de stabilité du gagnant réel (α=5) | [5,6] | [4,7] |
| Partial-inversion : le gagnant de la baseline survit dans son corridor ? | Non | Oui |
| Two-pass reproduit le gagnant de la baseline ? | Oui | Oui |
| Jaccard corridor (partial-inversion vs baseline) | 0.209 | 0.127 |

Sur les deux instances testées, `two-pass` s'avère fiable et rentable en temps de calcul. `partial-inversion` produit un gain de temps important mais son comportement vis-à-vis du vrai gagnant varie d'une instance à l'autre: un point à documenter comme limite de généralisation de cette heuristique plutôt qu'une propriété garantie.