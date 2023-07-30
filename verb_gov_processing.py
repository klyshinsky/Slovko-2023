from minio import Minio
from collections import defaultdict
from tqdm.auto import tqdm
import pickle
import pymorphy2
import time
from copy import deepcopy
import threading
from concurrent.futures.thread import ThreadPoolExecutor

# List of processed files.
files = [
'articles6_2.conllu', 
'detective_for_kidds.txt.conllu',
'articles_Vestnik_rayona.conllu', 'detective_masters.txt.conllu', 'habarahabr_2021_utf8.txt.conllu', 
'profile_ru.txt.conllu',
'dvnovosti.ru_khab.conllu', 'ibusines.txt.conllu', 'proza-ru_4.txt.conllu', 'topwar_2020_2.txt.conllu',
'chpsy.txt.conllu', 'dw-2021.txt.conllu', 'might_and_magic.txt.conllu', 'rbc-2020.txt.conllu', 'tsargrad-2021.txt.conllu',
'Foreign_Love_Stories.txt.conllu', 'mil_ru.txt.conllu', 'russian_action_fiction.txt.conllu', 'utro-2014.txt.conllu',
'commersant-2015.txt.conllu', 'russian_love_story.txt.conllu', 'vogue_2020_2.txt.conllu',
'compulenta-2013_short.txt.conllu', 'gazeta-4-ru-2020.txt.conllu', 'naked-science.txt.conllu', 
'Russian_UN.txt.conllu', 'vzglyad.txt.conllu',
'compulenta-2013.txt.conllu', 'nplus1_2020.txt.conllu', 'starhit-2021.txt.conllu', 'wikipedia_plain_text_2020_2.txt.conllu',
'gorky[ru]_2-2018.conllu', 'orcs_and_dwarfs.txt.conllu', 'swsys.txt.conllu',
'lenta_2018.txt.conllu', 'fontanka.txt.conllu', 'kosterin72ru.txt.conllu', 'membrana.txt.conllu'
]

#files = [
#    'wikipedia_plain_text_2020_2.txt.conllu'
#]

# Model of prepositional government for filtering wrong combinations by a grammatical case.
#  gent, datv, accs, ablt, loct
model = {
"О": (0.0, 0.0, 0.0769, 0.0, 0.923),
"В": (0.0, 0.0, 0.276, 0.0, 0.712), # - Manually added gent
"ОБ": (0.0, 0.0, 0.0222, 0.0, 0.978),
"ПО": (0.0, 0.986, 0.00859, 0.0, 0.00515), # - Manually added gent
"БЕЗ": (0.986, 0.0, 0.0, 0.0, 0.0),
"С": (0.213, 0.0, 0.00242, 0.772, 0.0),
"НЕ В": (0.0, 0.0, 0.288, 0.0, 0.712),
"ПРО": (0.0, 0.0, 1.0, 0.0, 0.0),
"ДО": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПРИ": (0.0, 0.0, 0.0, 0.0, 0.995),
"К": (0.0, 1.0, 0.0, 0.0, 0.0),
"СО": (0.321, 0.0, 1.0, 0.666, 0.0),
"НА": (0.0, 0.0, 0.358, 0.0, 0.635), # - Manually added gent
"ВО": (0.0, 0.0, 0.142, 0.0, 0.811),
"ЗА": (0.0, 0.0, 0.358, 0.603, 0.0),
"ПОД": (0.0, 0.0, 0.191, 0.779, 0.0),
"ПЕРЕД": (0.0, 0.0, 0.0, 1.0, 0.0),
"ДЛЯ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ЧЕРЕЗ": (0.0, 0.0, 1.0, 0.0, 0.0), # Manually added accs
"ПОСЛЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"МЕЖДУ": (0.156, 0.0, 0.0, 0.844, 0.0),
"НЕ БЕЗ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ВО ВРЕМЯ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В ХОДЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПРОТИВ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПРЕД": (0.0, 0.0, 0.0, 1.0, 0.0),
"НАД": (0.0, 0.0, 0.0, 1.0, 0.0),
"ВВИДУ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ОТНОСИТЕЛЬНО": ( 1.0, 0.0, 0.0, 0.0, 0.0), # -
"КО": (0.0, 1.0, 0.0, 0.0, 0.0),
"СООТВЕТСТВЕННО": ( 0.0, 1.0, 0.0, 0.0, 0.0), # -
"ОТ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В РЕЗУЛЬТАТЕ": ( 0.973, 0.0, 0.0, 0.0, 0.0),
"В ТЕЧЕНИЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"С ПОМОЩЬЮ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПУТЕМ": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"СОГЛАСНО": (0.0, 0.909, 0.0, 0.0, 0.0), # -
"В КАЧЕСТВЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ОКОЛО": (1.0, 0.0, 0.0, 0.0, 0.0),
"БЛАГОДАРЯ": (0.0, 1.0, 0.0, 0.0, 0.0),
"В ВИДЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В СООТВЕТСТВИИ С": ( 0.0, 0.0, 0.0, 1.0, 0.0),
"ПРИ ПОМОЩИ": (1.0, 0.0, 0.0, 0.0, 0.0),
"СРЕДИ": (1.0, 0.0, 0.0, 0.0, 0.0),
"МИМО": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"СКВОЗЬ": (0.0, 0.0, 1.0, 0.0, 0.0), # -
"ВДОЛЬ": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"МЕЖ": (0.0, 0.0, 0.0, 1.0, 0.0),
"РЯДОМ С": (0.0, 0.0, 0.0, 1.0, 0.0),
"В СТОРОНУ": (0.857, 0.0, 0.0, 0.0, 0.0),
"ВМЕСТЕ С": (0.0, 0.0, 0.0, 1.0, 0.0),
"ВПЕРЕДИ": (0.75, 0.0, 0.0, 0.0, 0.0), # -
"ПО НАПРАВЛЕНИЮ К": ( 0.0, 1.0, 0.0, 0.0, 0.0),
"ВСЛЕД ЗА": (0.0, 0.0, 0.0, 1.0, 0.0),
"ВОКРУГ": (0.833, 0.0, 0.0, 0.0, 0.0), # -
"СЛЕДОМ ЗА": (0.0, 0.0, 0.0, 1.0, 0.0),
"В НОГУ С": (0.0, 0.0, 0.0, 1.0, 0.0),
"НАПЕРЕКОР": (0.0, 1.0, 0.0, 0.0, 0.0),
"ВПЕРЕД": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"РАДИ": (1.0, 0.0, 0.0, 0.0, 0.0),
"НА ОСНОВАНИИ": ( 1.0, 0.0, 0.0, 0.0, 0.0),
"С УЧЕТОМ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В СВЯЗИ С": (0.0, 0.0, 0.0, 1.0, 0.0),
"ОТ ИМЕНИ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В НАЧАЛЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"У": (0.991, 0.0, 0.0, 0.0, 0.0),
"НА ПОРОГЕ": (0.833, 0.0, 0.0, 0.0, 0.0),
"ВНУТРИ": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"В ПРОЦЕССЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"НЕДАЛЕКО ОТ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ВБЛИЗИ": (1.0, 0.0, 0.0, 0.0, 0.0),
"НА МЕСТЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ВНЕ": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"НИЖЕ": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"НАПРОТИВ": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"ДАЛЕКО ЗА": (0.0, 0.0, 0.0, 1.0, 0.0),
"В ПРЕДЕЛАХ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ВОЗЛЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В РАСПОРЯЖЕНИИ": ( 1.0, 0.0, 0.0, 0.0, 0.0),
"ВЫШЕ": (1.0, 0.0, 0.0, 0.0, 0.0), # - 
"ПОСРЕДИ": (1.0, 0.0, 0.0, 0.0, 0.0),
"НАКАНУНЕ": (0.5, 0.0, 0.0, 0.0, 0.0), # -
"ПОСРЕДСТВОМ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ВОПРЕКИ": (0.0, 1.0, 0.0, 0.0, 0.0),
"ИЗ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ИЗО": (1.0, 0.0, 0.0, 0.0, 0.0),
"НА СТРАЖЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПОСЕРЕДИНЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В СВЕТЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"НЕПОДАЛЕКУ ОТ": ( 1.0, 0.0, 0.0, 0.0, 0.0),
"ПОСРЕДИНЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПОЗАДИ": (0.5, 0.0, 0.0, 0.0, 0.0),
"ПОВЕРХ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ВСЛЕД": (0.0, 1.0, 0.0, 0.0, 0.0),
"ЛИЦОМ К ЛИЦУ С": ( 0.0, 0.0, 0.0, 1.0, 0.0),
"В СИЛУ": (1.0, 0.0, 0.0, 0.0, 0.0),
"СОВМЕСТНО С": (0.0, 0.0, 0.0, 1.0, 0.0),
"ЗА СЧЕТ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ЗА СЧЁТ": (1.0, 0.0, 0.0, 0.0, 0.0),
"СПУСТЯ": (0.0, 0.0, 1.0, 0.0, 0.0), # -
"НА БЛАГО": (1.0, 0.0, 0.0, 0.0, 0.0),
"В ОБЛАСТИ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПО ОКОНЧАНИИ": ( 1.0, 0.0, 0.0, 0.0, 0.0),
"НЕ ЗА": (0.0, 0.0, 1.0, 1.0, 0.0),
"ВНИЗУ": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"ВМЕСТО": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПОПЕРЕК": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"ВНУТРЬ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В ПАМЯТЬ О": (0.0, 0.0, 0.0, 0.0, 1.0),
"ВПЛОТНУЮ К": (0.0, 1.0, 0.0, 0.0, 0.0),
"НА ПРОТЯЖЕНИИ": ( 1.0, 0.0, 0.0, 0.0, 0.0),
"В СОГЛАСИИ С": ( 0.0, 0.0, 0.0, 1.0, 0.0),
"БЕЗО": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПО СЛУЧАЮ": (1.0, 0.0, 0.0, 0.0, 0.0),
"СО СТОРОНЫ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ВСЛЕДСТВИЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В РАСПОРЯЖЕНИЕ": ( 1.0, 0.0, 0.0, 0.0, 0.0),
"С ЦЕЛЬЮ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В ШАГЕ ОТ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПО ОТНОШЕНИЮ К": ( 0.0, 1.0, 0.0, 0.0, 0.0),
"В РОЛИ": (1.0, 0.0, 0.0, 0.0, 0.0),
"СООТВЕТСТВЕННО С": ( 0.0, 0.0, 0.0, 1.0, 0.0),
"СВЕРХУ": (1.0, 0.0, 0.0, 0.0, 0.0), # -
"В ИНТЕРЕСАХ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ПО МЕРЕ": (1.0, 0.0, 0.0, 0.0, 0.0),
"НА СМЕНУ": (0.0, 1.0, 0.0, 0.0, 0.0),
"ПО СРАВНЕНИЮ С": ( 0.0, 0.0, 0.0, 1.0, 0.0),
"ИСХОДЯ ИЗ": (1.0, 0.0, 0.0, 0.0, 0.0),
"В ЗАВИСИМОСТИ ОТ": ( 1.0, 0.0, 0.0, 0.0, 0.0),
"ПО ПОВОДУ": (1.0, 0.0, 0.0, 0.0, 0.0),
"ИЗНУТРИ": (1.0, 0.0, 0.0, 0.0, 0.0)
}

cases = {"Gen":0, "Dat":1, "Acc":2, "Ins":3, "Loc":4}

# Reads a sentence in Universal Dependencies format.
def readSentence(lines, pos):
    ''' Returns a sentence from the __lines__ list which are in UD format.
        A sentence starts from the line with number __pos__.
    '''
    sent = []
    while pos < len(lines) and lines[pos] != '' and lines[pos] != '\r':
        if lines[pos][0] != '#':
            sent.append(lines[pos][:-2].split("\t"))
        pos += 1
    pos += 1
    return sent, pos

def get_next_lines(filename, pos_in_file, file_offset, res_lines, pos, minioClient=None):
    """ The function reads __file_offset__ bytes from the position __pos_in_file__.
If __minioClient__ is not None, reads __filename__ from a Minio storage, else reads __filename__ from local drive.
It's supposed that the string __res_lines__ stores the unused part of the srting after position __pos__. 
So, it copies the rest of the string at 0 position and reads the rest bytes until __file_offset__ len.
!!! For Minio, works with /public/syntax-parsed/ !!!
    """
    if minioClient:
        res_p = minioClient.get_object('public', 
                                       'syntax-parsed/'+filename, 
                                       pos_in_file, 
                                       file_offset).data
    else:
        with open(filename, 'rb') as in_file:
            in_file.seek(pos_in_file)
            res_p = in_file.read(file_offset)

    pos_in_file += file_offset
    res_lines = res_lines[pos:]
    success = False
    start = 0
    finish = file_offset
    
    while not success:
        if start > 1000 or finish < file_offset-1000:
            lines = ''
            finish = file_offset
            break
        try:
            lines = res_p[start:finish].decode('utf8')
            success = True
        except UnicodeDecodeError as err:
            if err.args[-1] == 'unexpected end of data':
                finish = err.args[-3] + start
            elif err.args[-1] == 'invalid start byte':
                start += 1
            else:
                start += 1
                finish -= 1

    pos_in_file -= file_offset - finish
    
    if len(res_lines) != 0:
        lines = res_lines[-1] + lines
        res_lines = res_lines[:-1]
    res_lines.extend(lines.split('\n'))
    return res_lines, pos_in_file


def buildTree(sentence):
    """ Builds a tree from CONLLU format.
    """ 
    root = -1
    for word in sentence:
        word.append([])
    for word in sentence:
        if word[0] != word[6]:
            sentence[int(word[6])-1][-1].append(word)
        
    return root
    
def find_child_Ng2(word, pos, cas):
    """ Finds all children of a node with id=parent, which are in genetive case. """
    children = []
    for child in word[-1]:
        if child[3] == pos and cas in child[5]:
            children.append(child)
    return children

def find_NNg2(sentence, combinations, cas):
    """ Finds combinations 'noun+noun_Gen".
    """
    buildTree(sentence)
    for i, word in enumerate(sentence):
        if word[3] == "NOUN" and i < len(sentence)-1 and \
           sentence[i+1][3] != 'ADP' and sentence[i+1][3] != 'CCONJ':
            children = find_child_Ng2(word, 'NOUN', cas)
            if children != []:
                for child in children:
                    combinations[word[2]][child[2]] += 1
    
def find_AN2(sentence, combinations, pos1, pos2):
    """ Finds combinations 'adj+noun".
    """
    buildTree(sentence)
    for i, word in enumerate(sentence):
        if word[3] == pos1:
            for child in word[-1]:
                if child[3] == pos2:
                    combinations[word[2]][child[2]] += 1

def find_no(word, sentence):
    """ Checks if there is particle "не" at this verb or a connected auxulary verb.
    """
    if word[6] != '0' and word[6] != 0:
        par_pos = int(word[6]) - 1
        if sentence[par_pos][3] == 'VERB' or sentence[par_pos][3] == 'ADJ':
            for child in sentence[par_pos][-1]:
                if child[2] == 'не' and child[3] == 'PART':
                    return True
    
    for child in word[-1]:
        if child[2] == 'не' and child[3] == 'PART':
            return True
        if child[3] == 'VERB' and find_no(child, sentence):
            return True
    return False

def find_prep(word, pos):
    """ Checks if there is a preposition in childs.
    """
    for child in word[-1]:
        if child[3] == pos:
            return child
    return None

def find_num(word):
    """ Checks if there is a quantitative adverb or ordinal number.
    """
    for child in word[-1]:
        if child[3] == 'NUM':
            return True
        if child[3] == 'ADV':
            if child[2] in ['много', 'немного', 'несколько', 'больше', 'меньше', 'большой', 'достаточно', \
                            'столько', 'мало', 'немало', 'столько' \
                           ]:
                return True
    return False

def add_preps(word, words, depth=0):
    """ Joins words consisting of a compound preposition. 
    """
    if depth > 2:
        return
    words.append((word[0], word[1]))
    for child in word[-1]:
        add_preps(child, words, depth+1)

def join_prep(word):
    """ Checks if the current word sequence can be joined to a compound preposition.
    """
    for child in word[-1]:
        if child[3] == 'ADP':
            words = []
            add_preps(child, words)
            words = sorted(words, key=lambda x: x[0])
            prep = ' '.join([w[1] for w in words])
            return prep.upper()
    return None

def get_case(word):
    """ Returns the grammatical case of a word if there is any.
    """
    params = word[5].split("|")
    for param in params:
        if param.startswith("Case="):
            return param[-3:]
    return ""

def find_VN2(sentence, combinations, pos, model, cases):
    """ Finds combinations 'verb+noun".
    """
    verb_pos = "VERB"
    noun_pos = "NOUN"
    adp_pos = "ADP"
    buildTree(sentence)
    for i, word in enumerate(sentence):
        if word[3] == pos and ((pos == verb_pos and not find_no(word, sentence)) or (pos != verb_pos)):
            for child in word[-1]:
                if child[3] == noun_pos:
                    if find_num(child):# or find_no(child, sentence):
                        continue
                    cas = get_case(child)
                    prep = find_prep(child, adp_pos)
                    if prep == None:
                        combinations[word[2]]['_'][cas][child[2]] += 1
                    else:
                        joined_prep = join_prep(child)
                        if cas in cases.keys() and joined_prep in model.keys() and \
                                model[joined_prep][cases[cas]] != 0:
                            combinations[word[2]][joined_prep][cas][child[2]] += 1
                        else:
                            pass
    
# The list of probable classes from Pymorphy, which are used for predicted words.
predict_classes = [pymorphy2.units.by_analogy.UnknownPrefixAnalyzer,
                   pymorphy2.units.by_analogy.KnownPrefixAnalyzer,
                   pymorphy2.units.by_analogy.KnownSuffixAnalyzer, 
                   pymorphy2.units.KnownPrefixAnalyzer,
                   pymorphy2.units.UnknAnalyzer, 
                   pymorphy2.units.by_analogy.KnownSuffixAnalyzer.FakeDictionary,
                   pymorphy2.units.by_hyphen.HyphenatedWordsAnalyzer
                  ]

# Checks if a word is predicted. Cashes results.
predicted_cash = {}
def is_predicted(word, morpho):
    if word in predicted_cash.keys():
        return predicted_cash[word]
    parse_results = morpho.parse(word)
    for parse_result in parse_results:
        for r in parse_result.methods_stack:
            if (type(r[0]) in predict_classes) or \
               (type(r[0]) is tuple and type(r[0][0]) in predict_classes ):
                predicted_cash[word] = True
                return True

    predicted_cash[word] = False
    return False

def create_combinations(action):
    """ Finds verb+prep+case+noun and noun+prep+case+noun depending on the __action__ parameter.
    """
    if action == "VPN" or action == "NPN":
        return defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(int))))
    # For the rest.
    else:
        return defaultdict(lambda:defaultdict(int))
    
def update_freqs(dct, replaces):
    """ Updeta frequencies in a dictionary __dct__ for pairs of equal words in __replaces__.
    """
    for w1, w2 in replaces:
        if w2 not in dct.keys():
            dct[w2] = deepcopy(dct[w1])
        else:
            dct[w2] += dct[w1]
        del dct[w1]

# Removes hyphernation and non-breaking space from words, joins equal words and updates their frequencies.
def replace_0xad(comb_arrs, actions):

    for combinations, action in zip(comb_arrs, actions):
        replacements = []
        for word1, dict1 in tqdm(combinations.items()):
            if action == "VPN" or action == "NPN":
                for prep, dict2 in dict1.items():
                    for cas, dict3 in dict2.items():
                        d = [(word2, word2.replace('\xad', '')) for word2 in dict3.keys() if '\xad' in word2]
                        if d != []:
                            update_freqs(dict3, d)
                            combinations[word1][prep][cas] = dict3
            else:
                d = [(word2, word2.replace('\xad', '')) for word2 in dict1.keys() if '\xad' in word2]
                if d != []:
                    update_freqs(dict1, d)
                    combinations[word1] = dict1
            if '\xad' in word1:
                replacements.append(word1)
        
        for replacement in replacements:
            del combinations[replacement]

def filter_predicted(dict3, morpho):
    """ Filters out all predicted words.
    """
    d = defaultdict(int)
    for word2, freq in dict3.items():
         if not is_predicted(word2, morpho):
             d[word2] = freq
    return d

# Удаляем все слова, которых нет в морфологии и сохранаяем.
def replace_non_vocabulary(action, combinations, morpho):
    """ Filters out all predicted words and saves result in a file.
    """
    comb3 = create_combinations(action)
    for word1, dict1 in tqdm(combinations.items()):
        if is_predicted(word1, morpho):
            continue
        if action == "VPN" or action == "NPN":
            for prep, dict2 in dict1.items():
                for cas, dict3 in dict2.items():
                    d = filter_predicted(dict3, morpho)
                    if len(d.keys()) != 0:
                        comb3[word1][prep][cas] = d
        else:
            d = filter_predicted(dict1, morpho)
            if len(d.keys()) != 0:
                comb3[word1] = d
    return comb3
    
def save_combinations(comb3, out_pickle_filename, action):
    """ Saves combinations into a pickle file. Since Pickle can not work with functions,
we have to convert DefaultDictionary into a regular dictionary.
    """

    if action == "VPN" or action == "NPN":
        comb_v = dict()
        for verb, dict1 in comb3.items():
            ddict1 = dict()
            for prep, dict2 in dict1.items():
                ddict2 = dict()
                for cas, dict3 in dict2.items():
                    ddict3 = {noun:freq for noun, freq in dict3.items()}
                    ddict2[cas] = ddict3
                ddict1[prep] = ddict2
            comb_v[verb] = ddict1
    
        with open(out_pickle_filename, "wb") as file:
            pickle.dump(comb_v, file)
        
    else:
        comb2 = deepcopy(comb3)
        comb2 = dict(comb2)
        with open(out_pickle_filename, "wb") as file:
            pickle.dump(comb2, file)
    
def replace_and_save(comb_arrs, actions, p_filenames):
    """ Replace any dummy words and save the result.
    """
    replace_0xad(comb_arrs, actions)
    morpho = pymorphy2.MorphAnalyzer()
    for arr_no, (action, out_pickle_filename) in enumerate(zip(actions, p_filenames)):
        comb_arrs[arr_no] = replace_non_vocabulary(action, comb_arrs[arr_no], morpho)
        save_combinations(comb_arrs[arr_no], out_pickle_filename, action)


def synchro_freqs(dct1, dct2, action):
    """ Synchronises frequency in two dictionaries. 
Actually, correctly adds __dct2__ dictionary to __dct1__ depending to __action__.
    """ 
    if action == "VPN" or action == "NPN":
        for verb, prep_dict in dct2.items():
            for prep, cas_dict in prep_dict.items():
                for cas, noun_dict in cas_dict.items():
                    for noun, freq in noun_dict.items():
                        dct1[verb][prep][cas][noun] += freq
    else:
        for noun, adj_dict in dct2.items():
            for adj, freq in adj_dict.items():
                dct1[noun][adj] += freq

def collect_data(filename, locki):
    """ Main function which extracts combinations from the file __filename__. 
They are working in parallel, so they need __locki__ for synchronization.     
    """
    print(f'++> running {filename}')
    # List of possible actions: verb+prep+case+noun, noun+prep+case+noun, adj+noun, adv+verb, adv+adj.
    # Just to find them all, because running along files takes long time (more than 24 hours for me).
    my_comb_arrs = [create_combinations("VPN"),
                    create_combinations("NPN"),
                    create_combinations("AN"),
                    create_combinations("AV"),
                    create_combinations("AA")
                   ]
    my_model = deepcopy(model)
    my_cases = deepcopy(cases)
    minioClient = Minio('127.0.0.1:9000',
                        access_key='public',
                        secret_key='123',
                        secure=False)
    res_lines = []
    res_lines, pos_in_file = get_next_lines(filename, 
                                            0, file_offset, res_lines, 0,
                                            minioClient
                                           )
    pos = 0

    for i in range(1000000000):
        try:
            sent, pos = readSentence(res_lines, pos)
        except:
            break
        try:
            find_VN2(sent, my_comb_arrs[0], "VERB", my_model, my_cases)
            find_VN2(sent, my_comb_arrs[1], "NOUN", my_model, my_cases)
            find_AN2(sent, my_comb_arrs[2], 'NOUN', 'ADJ')
            find_AN2(sent, my_comb_arrs[3], 'VERB', 'ADV')
            find_AN2(sent, my_comb_arrs[4], 'ADJ', 'ADV')
        except:
            pass
        if pos > len(res_lines) - 200:
            try:
                res_lines, pos_in_file = get_next_lines(filename, 
                                                        pos_in_file, file_offset, res_lines, pos,
                                                        minioClient
                                                       )
            except:
                break
            pos = 0

    # They are blocked just for synchronization. The rest of the time they are free.
    locki.acquire()
    print(f'= saving results {filename}')
    for i in range(5):
        synchro_freqs(comb_arrs[i], my_comb_arrs[i], actions[i])
    replace_and_save(comb_arrs, actions, p_filenames)
    locki.release()
    print(f'<--- finished {filename}')

def process_files(files):
    """ Starts the simultaneous precessing. I've selected 6 threads since it is faster for our server.
    """
    locki = threading.Lock()
    with ThreadPoolExecutor(max_workers=6) as executor:
        for filename in files:
            executor.submit(collect_data, filename, locki)
        

# Lets go!
if __name__ == "__main__":
    file_offset = 100000

    # Creates lists for the final result.
    comb_arrs = [create_combinations("VPN"),
                 create_combinations("NPN"),
                 create_combinations("AN"),
                 create_combinations("AV"),
                 create_combinations("AA")
                ]

    # List of possible actions: verb+prep+case+noun, noun+prep+case+noun, adj+noun, adv+verb, adv+adj.
    # Remove any if you don't need it.
    actions = ["VPN", "NPN", "AN", "AV", "AA"]
    # Save results in files according to an action.
    p_filenames = ["tmp/comb_verb_noun.pickle", "tmp/comb_noun_noun.pickle", 
                   "tmp/comb_noun_adj.pickle", "tmp/comb_verb_adv.pickle", "tmp/comb_adj_adv.pickle"]

    # Process, save and store.
    process_files(files)

