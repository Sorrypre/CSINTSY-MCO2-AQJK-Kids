import model
import affixing
import pandas

def mark_ne(p, c, n):
    c1 = p.lower() == 'si'
    c2 = p.lower() == 'the' and c[0].isupper()
    c3 = p == 'The'
    c4 = p == 'Ang'
    return 'NE' if c1 or c2 or c3 or c4 else ''

def pdfy(word_list):
    df = pandas.DataFrame([], columns=['word', 'previous_word', 'next_word', 'is_ne'])
    if len(word_list) > 0:
        for i in range(0, len(word_list)):
            prevw = word_list[i-1] if i > 0 else ''
            currw = word_list[i]
            nextw = word_list[i+1] if i < len(word_list)-1 else ''
            df.loc[len(df)] = [currw, prevw, nextw, mark_ne(prevw, currw, nextw)]
        #for w in word_list:
        #    df.loc[len(df)] = [w]
        for f in range(1, len(model.feature_columns)):
            df[model.feature_columns[f]] = df.apply(model.get_feature(f), axis=1)
    return df

"""
# Test run
if __name__ == '__main__':
    print(pdfy([ 'I', '\'', 'm', 'actually', 'not', 'sure', 'kung', 'pano', 'to', 'gawin', 'e', ',', \
    'I', '\'', 'm', 'pretty', 'bago', '-', 'bago', 'pa', 'on', 'this', 'one' ]))
"""