from classes import Tconverter
from classes import IntGuess
from classes import RelateNumbers
from classes import EvenDivision
from classes import NumberSong

#%%
def common(L1,L2):
    
    seen = False
    
    for i in L1:
        
        for j in L2:
           if i == j:
               seen = True
               break
        
        if seen is False:
            return False;
        seen = False
    
    return True

print(common([1,2,3,2],[3,2,1]))    


#%%

def isPermutation(L1,L2):
    
    if len(L1) is not len(L2):
        return False
    
    new_L2 = L2.copy()
    
    for i in L1:
        if i not in new_L2:
            return False
        else: 
            new_L2.remove(i)
    
    return True

print(isPermutation([1,5,4,4,7],[4,4,1,5,4,6]))

#%%

dic = {
        'Smells Like Teen Spirit': 6,
        'Polly': 3,
        'Scentless Apprentice': 7,
        'Rape Me': 6,
        'Lithium': 9}
 
for x,y in dic.items():
    if y>5:
        print(x,y)
    
#%%

def invertDic(dic):
    revertedDic = {}
    
    for x,y in dic.items():
        revertedDic[y] = x
    
    return revertedDic

dic = {
        'Smells Like Teen Spirit': 6,
        'Polly': 3,
        'Scentless Apprentice': 7,
        'Rape Me': 6,
        'Lithium': 9}
print(invertDic(dic))



#%%

def sieve(n):
    
    primes = []
    for i in range(n):
        primes.append(False)
    
    


























