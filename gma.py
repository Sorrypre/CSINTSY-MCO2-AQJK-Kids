import affixing

fil_pronouns = [
    'ikaw', 'kaw', '\'kaw', 'ka', 'mo', 'iyo', 'sayo', 'sa\'yo',
    'ako', 'ko', 'akin', 'sakin', 'sa\'kin',
    'siya', 'sya', 's\'ya', 'niya', 'nya', 'n\'ya', 'kaniya', 'kanya', 'kan\'ya', 'sakaniya', 'sakanya', 'sakan\'ya',
    'sila', 'nila', 'kanila', 'sakanila',
    'kayo', 'kamo', 'ninyo', 'niyo', 'inyo', 'sainyo',
    'tayo', 'natin', 'atin', 'satin', 'sa\'tin',
    'kami', 'namin', 'amin', 'samin', 'sa\'min',
    
    'ito', 'eto', '\'to', 'heto', 'nito', 'neto', 'dito', 'nandito', 'narito', 'ganito', 'ganto', 'gan\'to', 'ganire', 'gan\'re',
    'iyan', 'yan', '\'yan', 'hayan', 'niyan', 'nyan', 'n\'yan', 'diyan', 'dyan', 'd\'yan', 'jan', '\'jan',
        'nandiyan', 'nandyan', 'nand\'yan', 'nanjan', 'nan\'jan', 'ganiyan', 'ganyan', 'gan\'yan', 'gay-an', 'gay\'an', 'gayan',
    'iyon', 'yon', '\'yon', 'yun', '\'yun', 'yaon', 'hayon', 'niyon', 'nyon', 'n\'yon', 'noon', '\'nun', '\'non', 'doon', 'don', '\'don', 'dun', '\'dun',
        'nandoon', 'nandon', 'nando\'n', 'nandun', 'nandu\'n', 'naroon', 'naron', 'naro\'n', 'narun', 'naru\'n', 'ayon', 'ayun', '\'are', 'nare',
    
    'ano', 'sino', 'saan', 'san', 'sa\'n', 'alin', 'gaano', 'gano', 'ga\'no', 'ilan', 'bakit', 'paano', 'pano', 'pa\'no', 'kailan', 'kelan',
    'anuman', 'sinoman', 'sinuman', 'saanman', 'sanman', 'sa\'nman', 'alinman', 'gaanoman', 'ganuman', 'ganoman', 'ga\'numan', 'ga\'noman', 'ilanman', 'paanoman', 'panuman', 'panoman',
        'pa\'numan', 'pa\'noman', 'kailanman', 'kelanman'
]

eng_pronouns = [
    'I', 'i', 'me', 'myself', 'mine', 'my',
    'we', 'us', 'ourself', 'ourselves', 'ours', 'our',
    'you', 'u', 'yourself', 'urself', 'yourselves', 'urselves', 'yours', 'urs', 'your', 'ur',
    'thou', 'thee', 'thyself', 'theeself', 'thine',
    'yall', 'yallselves',
    'he', 'him', 'himself', 'hisself', 'his',
    'she', 'her', 'herself',
    'it', 'its', 'itself',
    'they', 'them', 'themselves', 'their', 'theirs',
    'one', 'oneself',
    
    'this', 'that', 'these', 'those', 'such',
    'all', 'any', 'every', 'everyone', 'everybody', 'somebody', 'anybody', 'someone', 'anyone', 'everyone',
    'noone', 'nothing', 'none', 'nobody',
    'who', 'whom', 'what', 'where', 'when', 'why', 'how',
    'whoever', 'whomever', 'whatever', 'wherever', 'whenever', 'however'
]

fil_first_clusters = [
    'gl', 'kl', 'pl',
    'br', 'dr', 'gr', 'pr', 'tr',
    'bw', 'kw', 'gw', 'pw', 'sw',
    'ky', 'dy', 'gy', 'ny', 'py', 'ty'
]

fil_middle_clusters = [
    'aa', 'ii', 'uo', 'ao',
    'kb', 'db', 'gb', 'lb', 'sb', 'tb', 'wb', 'yb',
    'ch', 'dk', 'gk', 'yk',
    'bd', 'kd', 'gd', 'md', 'pd', 'sd', 'td', 'yd',
    'sg', 'tg', 'wg', 'yg',
    'kh', 'dh', 'lh', 'mh', 'nh', 'yh',
    'kl', 'ml',
    'kp', 'dp', 'gp', 'mp', 'lp', 'wp',
    'ks', 'gs', 'ts', 'ws',
    'ry', 'rl', 'rg', 'rm',
    'bt', 'kt', 'gt', 'wt'
]

eng_first_clusters = [
    'aa', 'ae', 'ai', 'ao', 'au',
    'ea', 'ee', 'ei', 'eo', 'eu',
    'io',
    'oa', 'oi', 'oe', 'oo', 'ou',
    
    'bl', 'br', 'by',
    'cl', 'cr', 'cz', 'cy',
    'dr', 'dw', 'dy',
    'fl', 'fr', 'fj', 
    'gh', 'gl', 'gn', 'gr', 'gw', 'gy',
    'kn', 'kr',
    'ph', 'pl', 'pn', 'pr', 'ps', 'pt', 'py'
    'rh', 'ry',
    'sc', 'sf', 'sh', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sv', 'sw',
    'th', 'tr', 'ts', 'tw',
    'wh', 'wr', 'wy',
    'xy',
    'zh', 'zy'
]

eng_middle_clusters = [
    'ae', 'au', 'oo', 'ou', 'uu', 'ee', 'ea', 'ei', 'eo', 'eu',
    'bb', 'cc', 'dd', 'ff', 'gg', 'll', 'mm', 'nn', 'pp', 'rr', 'ss', 'tt', 'vv', 'xx', 'zz',
    'mb', 'mp', 'mf', 'nd', 'nk', 'ns', 'nt', 'nf', 'ns', 'nz', 'mn',
    'lb', 'lc', 'ld', 'lf', 'lg', 'lk', 'lm', 'lp', 'ls', 'lt', 'lv',
    'rb', 'rc', 'rd', 'rf', 'rg', 'rk', 'rm', 'rn', 'rp', 'rs', 'rt', 'rv',
    'sc', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sw',
    'ct', 'pt', 'bd',
    'ft', 'fs',
    'bl', 'br', 'cl', 'cr', 'dr', 'dl', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'tr', 'tl',
    'ch', 'gh', 'ph', 'sh', 'th', 'wh', 'zh',
    'ps', 'ts', 'ks', 'xt', 'dv', 'dj', 'gn'
]

def return_or_empty_if_eng(word):
    low = word.lower()
    if len(low):
        for p in eng_pronouns:
            if p == low:
                return p
    return ''
    
def return_or_empty_if_fil(word):
    low = word.lower()
    if len(low):
        for p in fil_pronouns:
            if p == low:
                return p
    return ''

def fil_first_cluster(word):
    if len(word) >= 3:
        lword = word.lower()
        first_str = lword[0:2]
        for f in fil_first_clusters:
            if f == first_str:
                return f
    return ''

def fil_middle_cluster(word):
    clusters = set()
    if len(word) >= 4:
        lword = word.lower()
        mid_strs = [lword[1:len(word)-1], lword[2:len(word)-1], lword[1:len(word)-2], lword[2:len(word)-2]]
        for s in mid_strs:
            for m in fil_middle_clusters:
                if m in s:
                    clusters.add(m)
    return ",".join(clusters)

"""
def fil_clusters(word):
    clusters = [[], [], []]
    lword = word.lower()
    # There can be only one cluster formed at the beginning,
    # i.e. the first two letters of the word
    first_str = lword[0:2]
    for f in fil_first_clusters:
        if f == first_str:
            clusters[0].append(f)
            break
    # To check for middle clusters, the word must be at least 4 letters long,
    # because there would be only 1 letter left in the middle if the word only has
    # 3 letters. (much worse for a 2-letter and a 1-letter word)
    if len(word) < 4:
        return clusters
    # The middle cluster could be anywhere in the middle of the word
    # The following has to be checked:
    # - word with first and last letter removed
    # - word with first two letters removed, as well as the last letter
    # - word with first letter as well as two last letters removed
    # - word with the first and last two letters removed
    mid_strs = [lword[1:len(word)-1], lword[2:len(word)-1], lword[1:len(word)-2], lword[2:len(word)-2]]
    for s in mid_strs:
        for m in fil_middle_clusters:
            if s.contains(m):
                clusters[1].append(m)
    # There are no actual C-C end clusters in Filipino words (unless conjugated or borrowed)
    return clusters
"""

def eng_first_cluster(word):
    if len(word) >= 3:
        lword = word.lower()
        first_str = lword[0:2]
        for f in eng_first_clusters:
            if f == first_str:
                return f
    return ''

def eng_middle_cluster(word):
    clusters = set()
    if len(word) >= 4:
        lword = word.lower()
        mid_strs = [lword[1:len(word)-2], lword[2:len(word)-2]]
        for s in mid_strs:
            for m in eng_middle_clusters:
                if m in s:
                    clusters.add(m)
    return ",".join(clusters)
    
def eng_end_cluster(word):
    if len(word) >= 3:
        lword = word.lower()
        end_str = lword[len(word)-2:]
        if all('a' <= c <= 'z' and c not in ['a', 'e', 'i', 'o', 'u'] for c in end_str):
            return end_str
    return ''

if __name__ == '__main__':
    print(eng_end_cluster('though'))

"""
def eng_clusters(word):
    clusters = [[], [], []]
    lword = word.lower()
    first_str = lword[0:2]
    for f in eng_first_clusters:
        if f == first_str:
            clusters[0].append(f)
            break
    if len(word) >= 4:
        # Since the consonant clusters in English is very complex,
        # we will only do (1,2) and (2,2)
        mid_strs = [lword[1:len(word)-2], lword[2:len(word)-2]]
        for s in mid_strs:
            for m in eng_middle_clusters:
                if s.contains(m):
                    clusters[1].append(m)
    # If it ends on a C-C, it's most likely English
    end_str = lword[len(word)-3:]
    if end_str[0] in affixing.consonants and end_str[1] in affixing.consonants:
        clusters[2].append(end_str)
    return clusters
"""