import csv
import pandas
csv.field_size_limit(2147483647)

def csv_as2d(fp_csv):
    entries = []
    if not fp_csv.endswith('.csv'):
        return entries
    with open(fp_csv) as target:
        rd = csv.reader(target, delimiter=',')
        for i, j in enumerate(rd):
            entries.append(j);
    return entries
    
def list_aspd(entries):
    # Dapat meisang entry, kahit column names lang
    # Otherwise wala akong magagawa sa parameter na yan
    if not len(entries):
        return None
    # Wag isasama ung column names, i.e. entries[0]
    # Kung walang laman kundi column names, edi empty list lang to
    ecpy = entries[1:] if len(entries) > 1 else []
    # Pwede na iconvert into pandas dataframe with columns as entries[0]
    data = pandas.DataFrame(ecpy, columns=entries[0])
    return data
    
def csv_makepd(fp_csv):
    return list_aspd(csv_as2d(fp_csv))

"""
# Test run
if __name__ == '__main__':
    data = csv_makepd('final_annotations.csv')
"""
