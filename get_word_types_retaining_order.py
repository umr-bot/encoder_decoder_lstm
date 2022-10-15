"""Get word type lists out of two parralel word token lists """
def get_type_lists(list_1,list_2):
    """Get word type lists out of two parralel word token lists"""
    tups = list(zip(list_1,list_2)) # pair tokens in list_1 and list_2 according to indices
    tups_set = list(set(tups)) # filter only unique token tuples

    t_set1,t_set2 = [list(t) for t in zip(*tups_set)] # unpack lists in tup_set
    return t_set1,t_set2
