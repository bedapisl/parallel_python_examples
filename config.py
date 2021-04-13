import os


def searched_words():
    keywords_virus = ['coronavirus', 'covid', 'covid-19', 'covid19', 'covid_19', 'covid 19', 'corona virus', 'covid2019', 'covid 2019', 'covid-2019', 'chinesevirus19', 'sars-cov-2', 'koronavir', 'korona vir', 'kowonaviris', 'coronavirussen', 'koronavirus', 'orthocoronavirinae', 'virus korona', 'koronabirus', 'kurunawirus', 'ruscorona', 'coronafirws', 'koronawirusy', 'koroonaviirused', 'koronavirusi', 'virusi vya corona', 'kronvirusoj']
    related_keywords = ['pandemi', 'flattenthecurve', 'coronaheros', 'stopthespread', 'coronalockdown', 'socialdistancing', 'social distancing', 'yomequedoencasa', 'quedateencasa', 'coronavirusoutbreak', 'stayathome', 'stayhome', 'quedateencasa', 'face mask', 'facemask', 'staythefuckhome', 'staythefuckathome', 'quarantena', 'quarantine', 'loveisnotcancelled', 'loveisnotcanceled', 'endthelockdown', 'postponedontcancel', 'covidlockdown', 'lockdown2020']
    keywords = keywords_virus + related_keywords 
    return keywords + [keyword.upper() for keyword in keywords]


def searched_files():
    return [os.path.join('data', filename) for filename in ['instagram.txt', 'facebook.txt', 'twitter.txt']]
