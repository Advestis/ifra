from ruleskit import RuleSet, ClassificationRule, RegressionRule


def apply_diff_privacy_classif(ruleset: RuleSet):
    pass


def apply_diff_privacy_regression(ruleset: RuleSet):
    pass


def apply_diff_privacy(ruleset: RuleSet):
    """
    Modifies rules predictions to make the learning private.
    P(pred(D1) in S) <= exp(epsilon * P(pred(D2) in S)) with D1 and D2 being different of only one point

    Parameters
    ----------
    ruleset


    Returns
    -------

    """
    if issubclass(ruleset.rule_type, ClassificationRule):
        apply_diff_privacy_classif(ruleset=ruleset)
    elif issubclass(ruleset.rule_type, RegressionRule):
        apply_diff_privacy_regression(ruleset=ruleset)
    else:
        raise TypeError(f"Invalid rule type {ruleset.rule_type}. Must derive from ClassificationRule or RegressionRule")
