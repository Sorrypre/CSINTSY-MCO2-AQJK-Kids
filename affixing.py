vowels = [
    'A', 'E', 'I', 'O', 'U',
    'a', 'e', 'i', 'o', 'u' 
]
consonants = [
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
    'ado',
    'an', 'han', 'nan',
    'ana',
    'ano',
    'ante',
    'aryo', 'ariyo',
    'asyon', 'asiyon',
    'ay',
    'en',
    'enyo', 'eño',
    'eriya', 'erya',
    'ero', 'era',
    'ibo', 'iba',
    'iko', 'hiko',
    'ilyo', 'illo',
    'ilya', 'illa',
    'in', 'hin', 'nin',
    'ing',
    'ismo', 'isma',
    'isto', 'ista',
    'isyo', 'isiyo',
    'ito', 'ita',
    'iyo', 'iya',
    'nayan',
    'ng',
    'og',
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

def relax_word(word):
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
        if not l in vowels and l in consonants:
            cc += 1
            if not first:
                first = True
        if l in vowels and first:
            consecs.append([cc, lpos])
            cc = 0
            first = False
        if l in vowels and not first:
            lpos = cpos + 1
        cpos += 1
        if cpos == len(strg) and cc > 0:
            consecs.append([cc, lpos])
            cc = 0
            first = False
    return consecs

def trim_prefix(word):
    lw = relax_word(word)
    if len(lw) >= len(_fil_prefix_list[len(_fil_prefix_list)-1]):
        for p in _fil_prefix_list:
            if len(lw) > len(p) and lw.startswith(p):
                return [p, lw[len(p):]]
    return ['', lw]

def trim_suffix(word):
    lw = relax_word(word)
    if len(lw) >= len(_fil_suffix_list[len(_fil_suffix_list)-1]):
        for s in _fil_suffix_list:
            if len(lw) > len(s) and lw.endswith(s):
                return [s, lw[:len(lw)-len(s)]]
    return ['', lw]
    
def trim_infix(word):
    low = relax_word(word)
    pref = trim_prefix(low)
    lw = pref[1]
    cc = consecutive_consonants(lw)
    # - the starting index of the left side of the infix
    #   will never exceed 2, i.e. the left side only has 3 letters
    if len(cc) > 0 and cc[0][1] <= 2:
        left = cc[0][1]
        right = cc[0][1]+cc[0][0]
        # * there are only two infixes of the same length of 2,
        #   but i just added this condition because well... dynamic programming hahaha
        # - the length of the word must be longer than the shortest infix
        if len(lw) >= len(_fil_infix_list[len(_fil_infix_list)-1]):
            for i in _fil_infix_list:
                # - the word should not start or end with the infix
                # - the word should not be the infix alone, therefore
                #   the length of the word must be longer than the infix
                if not lw.startswith(i) and not lw.endswith(i) and len(lw) > len(i):
                    if lw[right:right+len(i)] == i:
                        # Extracted letters left of the infix
                        exl = lw[left:right]
                        # Single extracted letter right of the infix
                        exr = lw[right+len(i)]
                        # - basically, C-I-C, C-C-I-C, V-I-C, and C-V-I-C patterns are not allowed
                        #   it should be C-I-V or C-C-I-V only in order to have a valid infix pattern
                        if all(c in consonants for c in exl) ^ (exr in consonants):
                            f = f"{lw[left:right]}{lw[right+len(i):]}"
                            return [i, f] if low == lw else [i, f"{pref[0]}{f}"]
    return ['', lw]

def on_prefix(word):
    return len(trim_prefix(word)[0]) > 0

def on_suffix(word):
    return len(trim_suffix(word)[0]) > 0

def on_infix(word):
    # Trim the prefix first to not duplicate the counting
    return len(trim_infix(trim_prefix(word)[1])[0]) > 0

def has_fil_affixing(word):
    return on_prefix(word) + on_suffix(word) + on_infix(word)

"""
# Test run
if __name__ == '__main__':
    print(has_fil_affixing('kulog'))
"""