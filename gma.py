import filAffixing

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
