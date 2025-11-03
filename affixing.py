"""
Note that these are not sorted lists.
It may seem sorted but it is really not.
"""

_vowels = [ 'a', 'e', 'i', 'o', 'u' ]
_fil_prefix_list = [
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
    'kawalang', 'kawalang-',
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
_fil_infix_list = [ 'in', 'um' ]
_fil_suffix_list = [
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
    'i',
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
    'syon', 'siyon',
]

def _relax_word(word):
    return word.lower()

def _on_prefix(word):
    relaxed = _relax_word(word)
    for pref in _fil_prefix_list:
        if relaxed.startswith(pref):
            return 1
    return 0

def _on_suffix(word):
    relaxed = _relax_word(word)
    for suff in _fil_suffix_list:
        if relaxed.endswith(suff):
            return 1
    return 0

def _on_infix(word):
    relaxed = _relax_word(word)
    for inf in _fil_infix_list:
        if relaxed[0] in _vowels and relaxed.startswith(inf) \
           or not relaxed[0] in _vowels and relaxed[1:].startswith(inf):
            return 1
    return 0
    
def has_fil_affixing(word):
    return _on_prefix(word) + _on_suffix(word) + _on_infix(word)
