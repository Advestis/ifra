from typing import List, Union, Tuple
from ruleskit import RuleSet
import logging

logger = logging.getLogger(__name__)


def adaboost_aggregation(
    rulesets: List[RuleSet], n_nodes: int, ruleset: Union[RuleSet, None], max_coverage: float
) -> Tuple[str, Union[RuleSet, None]]:
    """Among all rules generated by the nodes in the current iteration, will keep only the most recurent rule(s).

    Those rules will be added to  the given ruleset, returned by the function,
    and the points in each node's data activated by it will be ignored in order to find other relevant rules.
    In order not to remove too many points, a maximum of coverage is imposed.

    If among the rules extracted from the nodes, none are new compared to the current state of the model, nodes are
    not updated. If no new rules are found but each nodes had new data, learning is finished.

    Parameters
    ----------
    rulesets: List[RuleSet]
        New rulesets provided by the nodes.
    n_nodes: int
        Total number of nodes used in the learning
    ruleset: Union[RuleSet, None]
        Current central model ruleset to update. Can be None, in which case it will be created
    max_coverage: float
        Maximum coverage a node model's rule can have to be kept in the central model.

    Returns
    -------
    Tuple[str, Union[RuleSet, None]]
        First item of the tuple can be :\n
          * "updated" if central model was updates\n
          * "pass" if no new rules were found but learning should continue\n
          * "stop" otherwise.\n
        The second item is the aggregated ruleset if the first item is "updated", None otherwise
    """
    logger.info("Aggregating fit results using AdaBoost method...")
    all_rules = []
    for rs in rulesets:
        all_rules += rs.rules
    if len(all_rules) == 0:
        logger.info("... no rules found")
        if len(rulesets) == n_nodes:
            logger.info("No new rules were found, despite all nodes having provided a new model : learning is over")
            return "stop", None
        return "pass", None
    occurences = {r: all_rules.count(r) for r in set(all_rules) if r.coverage < max_coverage}
    if len(occurences) == 0:
        logger.warning("No rules matched coverage criterion")
        if len(rulesets) == n_nodes:
            logger.info("No new rules were found, despite all nodes having provided a new model : learning is over")
            return "stop", None
        return "pass", None
    max_occurences = max(list(occurences.values()))
    if ruleset is None:
        ruleset = RuleSet(
            [r for r in occurences if occurences[r] == max_occurences],
            remember_activation=False,
            stack_activation=False,
        )
    else:
        new_rules = [r for r in occurences if occurences[r] == max_occurences and r not in ruleset]
        if len(new_rules) == 0:
            logger.warning("No new rules found")
            if len(rulesets) == n_nodes:
                logger.info(
                    "No new rules were found, despite all nodes having provided a new model" " : learning is over"
                )
                return "stop", None
            return "pass", None
        ruleset += RuleSet(
            new_rules,
            remember_activation=False,
            stack_activation=False,
        )
    logger.info("... fit results aggregated")
    return "updated", ruleset
