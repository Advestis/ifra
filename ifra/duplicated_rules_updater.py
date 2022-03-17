import numexpr as ne
from collections import Counter
from typing import Optional, Union
import numpy as np
from ruleskit import RuleSet, ClassificationRule, RegressionRule, Rule
import logging

logger = logging.getLogger(__name__)


def get_weight(w: Union[None, str], rule: Rule):
    if w is None:
        return None
    if hasattr(rule, w):
        return getattr(rule, w)
    elif "rule." in w:
        for attr in dir(rule):
            if "rule." not in w:
                break
            if f"rule.{attr}" in w:
                value = getattr(rule, attr)
                if value is None:
                    raise ValueError(f"Attribute '{attr}' is not set")
                if np.isnan(value):
                    return None
                w = w.replace(f"rule.{attr}", str(getattr(rule, attr)))
        if "rule." in w:
            raise ValueError(f"Could not interprete all attributes specified in weight str : '{w}'")
        return ne.evaluate(w).reshape(1)[0]
    else:
        raise ValueError(f"I do not undersand the weight string '{w}'")


def update_duplicated_rules(
        aggregated_model: RuleSet,
        weight: Optional[str] = None,
        best: str = "max",
        name: Optional[str] = None
) -> bool:
    """Used by `ifra.aggragations.Aggregation.aggregate` to take into account the new aggregated model.

    It will update the rules predictions of duplicated rules. In case of classification rules, they become the most
    frequent prediction. In case of regression rules, they become the weighted mean of the predictions, using
    the rules attributes designed by 'weight'. This attributes are concidered better when they are taller if 'best'
    is "max", and better when they are smaller if 'best' is "min".

    It will set all rules attributes that are not the prediction to None, since theyr are only valid for a node's
    dataset.
    """

    attributes_to_keep = ["prediction"]

    if len(aggregated_model) == 0:
        logger.warning(f"{name} - No new rules in aggregated model")
        return False

    if aggregated_model.rule_type == ClassificationRule:
        success = update_classif_preds(aggregated_model, name=name)
    elif aggregated_model.rule_type == RegressionRule:
        success = update_regr_preds(aggregated_model, weight, best, name=name)
    else:
        raise TypeError(f"{name} - Unexpected model's rule type {aggregated_model.rule_type}")
    if not success:
        return False

    for attr in aggregated_model.rule_type.rule_index:
        if attr in attributes_to_keep:
            continue
        for r in aggregated_model:
            setattr(r, f"_{attr}", None)
    return True


def update_classif_preds(model: RuleSet, name: Optional[str] = None) -> bool:
    duplicated_conditions = {}
    to_remove = []
    conditions = [r.condition for r in model]

    if list(set(conditions)) == conditions:  # No duplicated conds -> no contradictory predictions -> do nothing
        return True

    for i in range(len(model)):
        if conditions.count(conditions[i]) > 1:
            if conditions[i] not in duplicated_conditions:
                duplicated_conditions[conditions[i]] = [i]
            else:
                duplicated_conditions[conditions[i]].append(i)

    for condition in duplicated_conditions:
        preds = []
        for i in duplicated_conditions[condition]:
            preds.append(model[i].prediction)
        if len(set(preds)) == 1:  # Duplicated conditions predict the same : nothing to do
            continue
        counter = Counter(preds).most_common(2)
        """Is a list of 2 tuples containing the 2 most frequent elements in *preds* with the number of times they
         are present"""
        if counter[0][1] == counter[1][1]:  # 2 or more different predictions are equally frequent : ignore rules
            to_remove += duplicated_conditions[condition]
            continue
        good_pred = counter[0][0]
        first = True
        for i in duplicated_conditions[condition]:
            # Only update first rule, and drop the others
            if first:
                # noinspection PyProtectedMember
                model[i]._prediction = good_pred
                first = False
            else:
                to_remove.append(i)

    model._rules = [model[i] for i in range(len(model)) if i not in to_remove]
    model.stack_activation = False
    model._activation = None
    model.stacked_activations = None
    model._coverage = None
    return True


def update_regr_preds(model: RuleSet, weight: str = "equi", best: str = "max", name: Optional[str] = None) -> bool:
    duplicated_conditions = {}
    to_remove = []
    conditions = [r.condition for r in model]

    if list(set(conditions)) == conditions:  # No duplicated conds -> no contradictory predictions -> do nothing
        return True

    for i in range(len(model)):
        if conditions.count(conditions[i]) > 1:
            if conditions[i] not in duplicated_conditions:
                duplicated_conditions[conditions[i]] = [i]
            else:
                duplicated_conditions[conditions[i]].append(i)

    for condition in duplicated_conditions:
        preds = []
        weights = []
        for i in duplicated_conditions[condition]:
            w = 1
            if weight != "equi":
                w = get_weight(weight, model[i])
            if w is None or np.isnan(w):
                continue
            preds.append(model[i].prediction)
            weights.append(w)
        weights = np.array(weights)
        weights = weights / weights.sum()
        if best == "min":
            weights = 1 - weights  # smaller weights become the best
            weights = weights / weights.sum()
        if weights.sum() == 0:
            logger.warning(f"{name} - Sum of weights is null. Discarding ruleset.")
            return False
        pred = np.average(np.array(preds), weights=weights)
        first = True
        for i in duplicated_conditions[condition]:
            # Only update first rule, and drop the others
            if first:
                # noinspection PyProtectedMember
                model[i]._prediction = pred
                first = False
            else:
                to_remove.append(i)

    model._rules = [model[i] for i in range(len(model)) if i not in to_remove]
    model.stack_activation = False
    model._activation = None
    model.stacked_activations = None
    model._coverage = None
    return True
