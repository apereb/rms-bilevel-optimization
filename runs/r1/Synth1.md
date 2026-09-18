# Synthèse — Instance 1 (period1) : baseline vs. heuristiques

## Configuration expérimentale

- **Instance** : `instance_period1.json` — α=5, ε=0.15, τ=3, λ=0.5, μ=0.4, δ=0.1, κ_add=1.5, κ_rem=1.0
- **Baseline** (`full`) : 3024 configurations, `time_limit=15s`
- **Two-pass** : passe 1 à 5s (toutes configs), passe 2 à 15s sur le sous-ensemble tel que `total_cost ≤ (1+1.5ε)×best_provisoire`
- **Partial-inversion** : conservation des 30% de configurations au Φ le plus élevé avant scheduling

## Résultats globaux

| Run | Configs planifiées | C* | Corridor | Gagnant | Coût du gagnant | Φ du gagnant |
|---|---|---|---|---|---|---|
| baseline | 3024 | 23.9781 | 526 | 1419 | 27.4424 | 3.1409 |
| two-pass | 3024 | 23.9781 | 527 | 1419 | 27.4424 | 3.1409 |
| partial-inversion | 907 | 24.3733 | 156 | 162 | 27.9333 | 3.1895 |

## Comparaisons deux-à-deux

| Paire | Jaccard (corridor) | Spearman (coût) | Kendall (coût) | Le 1er gagnant survit dans le corridor du 2nd | Écart de coût du gagnant du 2nd vs C* du 1er |
|---|---|---|---|---|---|
| baseline vs. two-pass | 0.998 | 0.999 | 0.979 | Oui | +14.45% |
| baseline vs. partial-inversion | 0.209 | 0.999 | 0.993 | Non | +16.50% |
| partial-inversion vs. two-pass | 0.209 | 0.998 | 0.975 | Non | +12.59% |

## Interprétation

### Two-pass

`two-pass` retrouve exactement le même gagnant que la baseline (`1419`) et le même coût total, avec un corridor quasi identique (Jaccard=0.998). Une passe grossière à 5s suffit à identifier la zone du corridor final, et ne raffiner que celle-ci à pleine précision (15s) retrouve la qualité d'un run intégralement résolu à 15s.

### Partial-inversion

`partial-inversion` ne se contente pas de sous-estimer la couverture du corridor (Jaccard=0.209 avec la baseline) — elle élit un gagnant différent (`162`), et ce gagnant n'appartient même pas au corridor de la baseline.

### Classement

Sur les trois paires comparées, Spearman ≥ 0.998 et Kendall ≥ 0.975 : le classement relatif des configurations communément évaluées par deux méthodes reste presque parfaitement préservé, quelle que soit la paire. La dégradation observée pour `partial-inversion` est donc une question de couverture (quelles configurations sont évaluées), pas de fiabilité du classement entre celles qui le sont.

### Sensibilité à α

Un balayage de α autour de la valeur utilisée (α=5) montre une forte sensibilité :

| α | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gagnant | 108 | 120 | 135 | 159 | 1419 | 1419 | 162 | 162 | 216 | 1743 | 1833 |

Le gagnant réel n'est stable que sur un intervalle étroit (α∈[5,6]) — presque chaque valeur d'α testée élit une configuration différente. Cette sensibilité doit être prise en compte lors du choix ou de la justification d'une valeur d'α : le résultat final n'est pas robuste à de petites variations de ce paramètre.

### Pareto-optimallité

Sur les 3024 configurations évaluées, seules 7 (0.23%) sont Pareto-optimales sur le plan (C_config, Q). Le gagnant retenu n'en fait pas partie : il est dominé sur le plan coût/débit pur par d'autres configurations. Ceci est cohérent avec la logique de sélection en deux phases — le gagnant est choisi par argmax Φ *à l'intérieur* du corridor de coût, non par optimalité conjointe (coût, débit).