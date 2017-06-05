import string

def genAlphaNum():
    i = 1
    while True:
        for c in string.lowercase:
            yield (str(i) + str(c))
        i += 1
