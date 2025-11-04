_vowels = [
    'A', 'E', 'I', 'O', 'U',
    'a', 'e', 'i', 'o', 'u' 
]
_consonants = [
    'B', 'K', 'D', 'G', 'H', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Y',
    'b', 'k', 'd', 'g', 'h', 'l', 'm', 'n', 'p', 'r', 's', 't', 'w', 'y'
]
_fil_raw_prefix_list = [
    'a', 'a-',
    'ag', 'ag-',
    'apaka', 'apaka-',
    'bilang', 'bilang-',
    'dalub', 'dalub-',
    'danum', 'danum-',
    'de', 'de-',
    'di', 'di-',
    'ga', 'ga-',
    'gaga', 'gaga-',
    'gi', 'gi-',
    'git', 'git-',
    'hi', 'hi-',
    'hing', 'hing-', 'him',
    'i', 'i-',
    'ika', 'ika-',
    'ikapag', 'ikapag-',
    'in', 'in-',
    'ipa', 'ipa-',
    'ipag', 'ipag-',
    'ipang', 'ipang-',
    'isa', 'isa-',
    'ka', 'ka-',
    'kaka', 'kaka-',
    'kasing', 'kasing-',
    'kawalang', 'kawalang-', 'kawalam',
    'ki', 'ki-',
    'kina', 'kina-',
    'kontra', 'kontra-',
    'laban', 'laban-',
    'labing', 'labing-',
    'ma', 'ma-',
    'mag', 'mag-',
    'maga', 'maga-',
    'magka', 'magka-',
    'magkaka', 'magkaka-',
    'magpa', 'magpa-',
    'magpaka', 'magpaka-',
    'magsi', 'magsi-',
    'mai', 'mai-', 'ma-i', 'ma-i-',
    'maka', 'maka-',
    'maki', 'maki-',
    'makipag', 'makipag-',
    'mala', 'mala-'
    'mang', 'mang-', 'mam', 'man',
    'mapa', 'mapa-',
    'mapag', 'mapag-',
    'mapang', 'mapang-',
    'may', 'may-',
    'mo', 'mo-', 'mu', 'mu-',
    'na', 'na-',
    'nai', 'nai-', 'na-i', 'na-i-',
    'nag', 'nag-',
    'naga', 'naga-',
    'nagka', 'nagka-',
    'nagkaka', 'nagkaka-',
    'nang', 'nang-', 'nam', 'nan',
    'nagsi', 'nagsi-'
    'nagsipag', 'nagsipag-',
    'naka', 'naka-',
    'nakaka', 'nakaka-',
    'nakakapag', 'nakakapag-',
    'naki', 'naki-',
    'nakiki', 'nakiki-',
    'napaka', 'napaka-',
    'ni', 'ni-',
    'ning', 'ning-',
    'pa', 'pa-',
    'pag', 'pag-',
    'paging', 'paging-'
    'pagiging', 'pagiging-',
    'pagka', 'pagka-',
    'pagkaka', 'pagkaka-',
    'pagkakapaging', 'pagkakapaging-',
    'pagkapagiging', 'pagkapagiging-',
    'paka', 'paka-',
    'paki', 'paki-',
    'pakiki', 'pakiki-',
    'pakikipag', 'pakikipag-',
    'pala', 'pala-',
    'pampa', 'pampa-',
    'pangpa', 'pangpa-', 'pang-pa', 'pang-pa-',
    'panag', 'panag-',
    'pang', 'pang-', 'pam', 'pan',
    'pina', 'pina-',
    'pinag', 'pinag-',
    'pinaka', 'pinaka-',
    'pinang', 'pinang-',
    'sa', 'sa-',
    'sali', 'sali-',
    'sang', 'sang-',
    'sari', 'sari-',
    'sing', 'sing-',
    'tag', 'tag-',
    'taga', 'taga-',
    'tagapag', 'tagapag-',
    'tali', 'tali-',
    'tig', 'tig-',
    'tiga', 'tiga-',
    'ting', 'ting-',
    'um', 'um-'
]
_fil_raw_infix_list = [ 'in', 'um' ]
_fil_raw_suffix_list = [
    'a', 'ha',
    'ado',
    'an', 'han', 'nan',
    'ana',
    'ano',
    'ante',
    'asyon', 'asiyon',
    'ay',
    'e',
    'en',
    'enyo', 'eño',
    'eriya', 'erya',
    'ero', 'era',
    'i', 'hi',
    'ibo', 'iba',
    'ilyo', 'illo',
    'ilya', 'illa',
    'in', 'hin', 'nin',
    'ing',
    'is', 'as', 'os', 'us',
    'ismo', 'isma',
    'isto', 'ista',
    'ito', 'ita',
    'iyo', 'iya',
    'nayan',
    'ng',
    'on',
    'ong', 'ang',
    'oy',
    'sya', 'siya',
    'syon', 'siyon'
]

"""
    Sort from longest to shortest affix
    Para ung mahahabang affixes muna ang itetest na may possible subset na ibang affix
    e.g. -a vs -ha, -in vs -hin
"""
_fil_prefix_list = sorted(_fil_raw_prefix_list, key=len, reverse=True)
_fil_infix_list = sorted(_fil_raw_infix_list, key=len, reverse=True)
_fil_suffix_list = sorted(_fil_raw_suffix_list, key=len, reverse=True)

def _relax_word(word):
    return word.lower()

"""
    Returns a list of consecutive consonants found within strg in the form 
    of an integer pair [consecutive_count, from_zero_based_index].
"""
def consecutive_consonants(strg):
    if not strg.strip():
        return []
    consecs = []
    cc = 0
    cpos = 0
    lpos = 0
    first = False
    for l in strg:
        if not l in _vowels and l in _consonants:
            cc += 1
            if not first:
                first = True
        if l in _vowels and first:
            consecs.append([cc, lpos])
            cc = 0
            first = False
        if l in _vowels and not first:
            lpos = cpos + 1
        cpos += 1
        if cpos == len(strg) and cc > 0:
            consecs.append([cc, lpos])
            cc = 0
            first = False
    return consecs

def _postprefix2(word):
    relaxed = _relax_word(word)
    for pref in _fil_prefix_list:
        if relaxed.startswith(pref):
            if len(word) > len(pref)+2:
                return word[len(pref):len(pref)+2]
            elif len(word) == len(pref):
                return ' '
            else:
                pass
    # No matches
    return ''

def _on_prefix(word):
    pp2 = _postprefix2(word)
    if not pp2.strip():
        return 0
    return 1 if consecutive_consonants(pp2)[0][0] < 2 else 0

def _presuffix3(word):
    relaxed = _relax_word(word)
    for suff in _fil_suffix_list:
        if word.endswith(suff):
            if len(word) > len(suff):
                return word[-len(suff)-3:-len(suff)]
            elif len(word) == len(suff):
                return ' '
            else:
                return ''
    # No matches
    return ''

def _on_suffix(word):
    ps3 = _presuffix3(word)
    if not ps3.strip():
        return 0
    return 1 if consecutive_consonants(ps3)[-1][0] < 3 else 0

def _on_infix(word):
    if len(word) < 4:
        return 0
    relaxed = _relax_word(word)
    first_two = False;
    if consecutive_consonants(relaxed[0:2])[0][0] == 2:
        if len(relaxed) < 5:
            return 0
        first_two = True
    # Filipino infix law: C-V -> C-I-V or C-C-V -> C-C-I-V only
    for inf in _fil_infix_list:
        L = 2 if first_two else 1
        R = L + len(inf)
        if relaxed[L:R] == inf and relaxed[R] in _vowels:
            return 1
    return 0

def has_fil_affixing(word):
    return _on_prefix(word) + _on_infix(word) + _on_suffix(word)

"""
if __name__ == '__main__':
    #for p in consecutive_consonants("r=ead"):
    #    print("%d %d"%(p[0],p[1]))
    print(has_fil_affixing("pinagkainan"))
"""