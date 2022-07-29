import json

params = {
    'N' : 40,
    'S' : 14000,
    'T' : 6000,
    'K' : 4,
    'f' : 0.1,
    'stepK' : 1,
    'binomial_p' : 0.4,
    'gaussian_stdK' : 1,
    'lognormal_stdK' : 1,
    'reps' : 3
        }

pfile = open('params.json', "w")
json.dump(params, pfile)
pfile.close()
