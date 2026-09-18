## Indices de comparaison

### Indice de Jaccard

**Indice de Jaccard** — mesure le recouvrement entre deux ensembles :

`|A∩B| / |A∪B|`

Vaut 1 si les deux ensembles sont identiques, 0 s'ils sont disjoints.

Dans nos comparaisons, on l'utilise pour quantifier à quel point le corridor retrouvé par une heuristique recouvre le vrai corridor de la baseline. Un Jaccard de `0.2` signifie que les deux corridors ne partagent qu'une petite fraction de leurs éléments, même s'ils ont chacun une taille raisonnable.

### Corrélation de rang de Spearman

**Corrélation de rang de Spearman (ρ)** — mesure si deux classements ordonnent les mêmes éléments dans le même ordre, sans exiger que les valeurs elles-mêmes soient proches.

Concrètement : on classe chaque configuration par coût dans les deux runs comparés, puis on corrèle ces deux classements (pas les coûts bruts).

Elle varie de -1 (ordre parfaitement inversé) à +1 (ordre identique), avec 0 signifiant aucune relation.

On l'utilise pour vérifier qu'une heuristique préserve la hiérarchie relative des configurations, même quand elle ne trouve pas les mêmes valeurs de coût exactes.

### Corrélation de rang de Kendall

**Corrélation de rang de Kendall (τ)** — répond à la même question que Spearman (les deux classements s'accordent-ils ?) mais par une méthode différente.

Elle compte la proportion de paires d'éléments classées dans le même ordre relatif dans les deux classements, moins celles classées à l'envers, normalisée par le nombre total de paires.

Elle est plus sensible aux petites perturbations locales du classement que Spearman, et son interprétation ('proportion de paires concordantes') est souvent plus intuitive pour un lecteur.

On rapporte les deux car elles peuvent diverger légèrement quand les erreurs de classement se concentrent différemment : nombreuses petites inversions vs. quelques grandes inversions.

## Pourquoi les trois ensemble ?

Les trois indicateurs répondent à des questions complémentaires :

- **Jaccard** mesure la proportion de configuration du corridor de la baseline restant.
- **Spearman / Kendall** regarde parmi celles en commun comment elles sont classées.

Une heuristique peut avoir un Jaccard très faible (peu de recouvrement) tout en ayant un Spearman/Kendall très élevé (l'ordre relatif de ce qu'elle voit reste fiable).

C'est exactement ce qu'on observe pour `partial-inversion` dans nos résultats : elle rate beaucoup de configurations, mais ne se trompe pas sur l'ordre de celles qu'elle évalue.