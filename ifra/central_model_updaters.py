from collections import Counter

from ruleskit import RuleSet, ClassificationRule, RegressionRule
import logging


logger = logging.getLogger(__name__)


def central_model_update(central_model: RuleSet, aggregated_model: RuleSet) -> bool:
    """Used by `ifra.central_server.CentralServer` to take into account the new aggregated model.

    It will ignore rules already present in the central model (even though there should be None, because ndoes would
    have already filtered duplicated rules out).
    Checks that aggregated model rule types are the same (regression or classification) compared to current central
    model.
    """

    if central_model.rule_type is not None and (central_model.rule_type != aggregated_model.rule_type):
        raise TypeError("Central model's and aggregated model's rules type are different")
    
    new_rules = 0
    for rule in aggregated_model:
        if rule not in central_model:
            new_rules += 1
            central_model.append(rule)
    
    if new_rules == 0:
        logger.warning("No new rules in aggregated model")
        return False
    
    if central_model.rule_type == ClassificationRule:
        update_classif_preds(central_model)
    elif central_model.rule_type == RegressionRule:
        update_regr_preds(central_model)
    else:
        raise TypeError(f"Unexpected model's rule type {central_model.rule_type}")
    return True


def update_classif_preds(model: RuleSet):
    """At this point, the central model can contain rules with identical conditions but different predictions.
    This method aims at solving that : for duplicated conditions, the corresponding rules will have their
    predictions set to be the one of the most recurring prediction. If several predictions are equally frequent, the
    rules corresponding to the duplicated conditions are discarded.

    Parameters
    ----------
    model: RuleSet
        Modified in place
    """
    duplicated_conditions = {}
    to_remove = []
    conditions = [r.condition for r in model]

    if list(set(conditions)) == conditions:  # No duplicated conds -> no contradictory predictions -> do nothing
        return

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
        for i in duplicated_conditions[condition]:
            # noinspection PyProtectedMember
            model[i]._prediction = good_pred

    model._rules = [model[i] for i in range(len(model)) if i not in to_remove]
    model.remember_activation = False
    model.stack_activation = False
    model._activation = None
    model.stacked_activations = None
    model._coverage = None


def update_regr_preds(self):
    # TODO (pcotte)
    pass
