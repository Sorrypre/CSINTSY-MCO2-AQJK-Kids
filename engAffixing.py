_eng_raw_prefix_list = ['a', 'an', 'ante', 'anti', 
'auto',
'circum', 
'co', 'com', 'con',
'contra','contro',
'de',
'dis',
'en',
'ex',
'extra',
'hetero',
'homo',
'homeo',
'hyper',
'il','im','in','ir',
'in','inter',
'intra','intro',
'macro',
'micro',
'mono',
'non',
'omni',
'post',
'pre','pro',
're',
'sub',
'sym','syn'
'tele',
'trans',
'tri',
'un',
'uni',
'up',
]
_eng_raw_suffix_list = ['acy',
'al',
'ance','ence',
'dom',
'er','or',
'ism',
'ist',
'ity','ty',
'ment',
'ness',
'ship',
'sion', 'tion',
'ate',
'en',
'ify', 'fy',
'ize','ise',
'able','ible',
'al', 
'esque',
'ful',
'ic','ical',
'ious','ous'
'ish','ive',
'less',
'y'
]

_eng_prefix_list = sorted(_eng_raw_prefix_list, key=len, reverse=True)
_eng_suffix_list = sorted(_eng_raw_suffix_list, key=len, reverse=True)

def relax_word(word):
    return word.lower()

def trim_prefix(word):
    lw = relax_word(word)
    if len(lw) >= len(_eng_prefix_list[len(_eng_prefix_list)-1]):
        for p in _eng_prefix_list:
            if len(lw) > len(p) and lw.startswith(p):
                return [p, lw[len(p):]]
    return ['', lw]

def trim_suffix(word):
    lw = relax_word(word)
    if len(lw) >= len(_eng_suffix_list[len(_eng_suffix_list)-1]):
        for s in _eng_suffix_list:
            if len(lw) > len(s) and lw.endswith(s):
                return [s, lw[:len(lw)-len(s)]]
    return ['', lw]

def on_eng_prefix(word):
    return len(trim_prefix(word)[0]) > 0

def on_eng_suffix(word):
    return len(trim_suffix(word)[0]) > 0

def has_eng_affixing(word):
    return on_eng_prefix(word) * 2 + on_eng_suffix(word)