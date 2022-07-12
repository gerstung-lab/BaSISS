import numpy as np

'''
Deprecated, remove later
'''
def name_exchange_necrfocal(x):
    if x['Necrosis'] == 'Present - Both' or x['Necrosis'] == 'Present - focal':
        return 'Present'
    elif x['Necrosis'] == 'Absent' or x['Necrosis'] == 'Present - Central':
        return 'Absent'
    else:
        return x['Necrosis']


def name_exchange_necrcentral(x):
    if x['Necrosis'] == 'Present - Both' or x['Necrosis'] == 'Present - Central':
        return 'Present'
    elif x['Necrosis'] == 'Absent' or x['Necrosis'] == 'Present - focal':
        return 'Absent'
    else:
        return x['Necrosis']


def name_exchange_necr(x):
    if x['Necrosis'] == 'Present - Both' or x['Necrosis'] == 'Present - Central' or x['Necrosis'] == 'Present - focal':
        return 'Present'
    elif x['Necrosis'] == 'Absent':
        return 'Absent'
    else:
        return x['Necrosis']


def name_exchange_lymph_stroma(x):
    if x['Lymphocyte infiltrate, stroma'] == 'Prominent' or x['Lymphocyte infiltrate, stroma'] == 'Subtle':
        return 'Present'
    elif x['Lymphocyte infiltrate, stroma'] == 'Absent':
        return 'Absent'
    else:
        return x['Lymphocyte infiltrate, stroma']


def name_exchange_lymph_tumor(x):
    if x['Lymphocyte infiltrate, tumor'] == 'Prominent' or x['Lymphocyte infiltrate, tumor'] == 'Subtle':
        return 'Present'
    elif x['Lymphocyte infiltrate, tumor'] == 'Absent':
        return 'Absent'
    else:
        return x['Lymphocyte infiltrate, stroma']


def name_exchange_anathomy(x):
    if np.isin(x['Anatomic structure'], ['Terminal lobuli', 'TDLU/Terminal lob.', 'TDLU',
                                         'TDLU/terminal lobuli']):
        return 'TDLU'
    elif x['Anatomic structure'] == 'Duct':
        return 'Duct'
    else:
        return x['Anatomic structure']


def name_exchange_growth(x):
    if np.isin(x['Growth pattern'], ['mixed', 'Mixed', 'IDC, mixed']):
        return 'Mixed'
    elif x['Growth pattern'] == 'IDC, stranded':
        return 'Stranded'
    elif x['Growth pattern'] is np.nan:
        return 'Absent'
    elif x['Growth pattern'] == 'Solid':
        return 'Solid'
    else:
        return x['Growth pattern']
