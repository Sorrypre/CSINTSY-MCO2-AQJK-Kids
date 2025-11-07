import model
import affixing
import pandas

def pdfy(word_list):
    df = pandas.DataFrame([], columns=[model.feature_columns[0]])
    if len(word_list) > 0:
        for w in word_list:
            df.loc[len(df)] = [w]
        for f in range(1, len(model.feature_columns)):
            df[model.feature_columns[f]] = df.apply(model.get_feature(f), axis=1)
    return df

"""
# Test run
if __name__ == '__main__':
    print(pdfy([ 'I', '\'', 'm', 'actually', 'not', 'sure', 'kung', 'pano', 'to', 'gawin', 'e', ',', \
    'I', '\'', 'm', 'pretty', 'bago', '-', 'bago', 'pa', 'on', 'this', 'one' ]))
"""