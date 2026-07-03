from poker_common import Card, parse, RANK_STR
from hand_rank import HandRank

def best_hand(seven : list[str]) -> tuple[HandRank, list[str]] :
    bit_all : int = 0
    bit_each : dict[str, int] = {}
    lst_each_value : dict[int, list[Card]] = {}
    
    cards = parse(seven)
    
    for card in cards :
        bit_all |= 1 << card.value
        bit_each[card.suit] = bit_each.get(card.suit, 0) | (1 << card.value)
        lst_each_value[card.value] = lst_each_value.get(card.value, []) + [card]
    

    def mask_n(st : int, n : int) -> int :
        return ((1 << n) - 1) << st
    def mask_5(st : int) -> int :
        return mask_n(st, 5)
    def single(v : int) -> int :
        return 1 << v
    def contain(S : int, pattern : int) -> bool :
        return (S & pattern) == pattern
    
    def make_card(v : int, suit : str) -> Card :
        return Card(RANK_STR[v - 2], suit)
    
    # check royal flush
    for suit, bit in bit_each.items() :
        if contain(bit, mask_5(10)) :
            return HandRank.ROYAL_FLUSH, \
                [str(make_card(v, suit)) for v in range(14, 9, -1)]
    
    # check straight flush
    for suit, bit in bit_each.items() :
        for st in range(10, 0, -1) :
            if contain(bit, mask_5(st)) :
                return HandRank.STRAIGHT_FLUSH, \
                    [str(make_card(v, suit)) for v in range(st + 4, st - 1, -1)]
    # special case for A2345 straight flush
    for suit, bit in bit_each.items() :
        if contain(bit, single(14) | mask_n(2, 4)) :
            return HandRank.STRAIGHT_FLUSH, \
                    [str(make_card(v, suit)) for v in [5, 4, 3, 2, 14]]
    
    # check four of a kind
    for v, lst in lst_each_value.items() :
        if len(lst) == 4 :
            kicker = max((c for lst in lst_each_value.values() for c in lst if c.value != v), 
                         key=lambda c: c.value, 
                         default=None)
            return  HandRank.FOUR_OF_A_KIND, \
                    [str(c) for c in lst] \
                    + ([str(kicker)])
    
    # check full house
    three = None
    pair = None
    for v, lst in sorted(lst_each_value.items(), 
                           key=lambda item: item[0], 
                           reverse=True) :
        if len(lst) == 3 and three is None :
            three = lst
        elif len(lst) >= 2 and pair is None :
            pair = lst[:2]
    if three is not None and pair is not None :
        return HandRank.FULL_HOUSE, [str(c) for c in three + pair]
    
    # check flush
    flush_cards = []
    for suit, bit in bit_each.items() :
        if bin(bit).count('1') >= 5 :
            flush_cards.append(sorted((c for c in cards if c.suit == suit), 
                                 key=lambda c: c.value, 
                                 reverse=True))
    
    if len(flush_cards) > 0 :
        flush_cards.sort(key=lambda lst: tuple([c.value for c in lst]), reverse=True)
        return HandRank.FLUSH, [str(c) for c in flush_cards[0][:5]]
    
    # check straight
    for st in range(10, 0, -1) :
        if contain(bit_all, mask_5(st)) :
            vls = [st + 4, st + 3, st + 2, st + 1, st]
            return HandRank.STRAIGHT, \
                [str(lst_each_value[v][0]) for v in vls]
    # special case for A2345 straight
    if contain(bit_all, single(14) | mask_n(2, 4)) :
        return HandRank.STRAIGHT, \
            [str(lst_each_value[v][0]) for v in [5, 4, 3, 2, 14]]
    
    # check three of a kind
    for v, lst in sorted(lst_each_value.items(), key=lambda item: item[0], reverse=True) :
        if len(lst) == 3 :
            kickers = sorted((c for c in cards if c.value != v), 
                             key=lambda c: c.value, 
                             reverse=True)
            return HandRank.THREE_OF_A_KIND, [str(c) for c in lst + kickers[:2]]
    
    # check two pair
    pairs = []
    for v, lst in sorted(lst_each_value.items(), key=lambda item: item[0], reverse=True) :
        if len(lst) >= 2 :
            pairs.append(lst[:2])
            if len(pairs) == 2 :
                break
    
    if len(pairs) >= 2 :
        lst = [c for c in cards if c.value != pairs[0][0].value and c.value != pairs[1][0].value]
        lst.sort(key=lambda c: c.value, reverse=True)
        return HandRank.TWO_PAIR, [str(c) for c in pairs[0] + pairs[1] + lst[:1]]
    
    # check one pair
    if len(pairs) == 1 :
        lst = [c for c in cards if c.value != pairs[0][0].value]
        lst.sort(key=lambda c: c.value, reverse=True)
        return HandRank.ONE_PAIR, [str(c) for c in pairs[0] + lst[:3]]
    
    # high card
    high_cards = sorted(cards, key=lambda c: c.value, reverse=True)
    return HandRank.HIGH_CARD, [str(c) for c in high_cards[:5]]
    