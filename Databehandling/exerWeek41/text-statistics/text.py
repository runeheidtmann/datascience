import re
import matplotlib


stopwords = open('stopwords').read().split("\n")

def statistics(url):
    file = open(url,"r")
    d = {}
    line = file.readline()
    while line:
        lower = line.lower()
        cleaned = re.sub('[^a-z\'\- ]','', lower)
        words = cleaned.strip().split(' ')
        for w in words:
            if w in d:
                d[w] += 1
            elif w not in stopwords:
                d[w] = 1
        line = file.readline()
    file.close()
    return d

def sentiment(url):
    
    count = 0
    words_in_text = statistics(url)
    
    negative_file = open("negative.txt", "r")
    negative_list = []
    
    for line in negative_file:
        negative_list.append(line.strip())

    
    positive_file = open("positive.txt", "r")
    positive_list = []
    
    for line in positive_file:
        positive_list.append(line.strip())
    
    
    
    for key,value in words_in_text.items():
        
        if key in negative_list:   
            print("Negative: "+str(count)+"-"+str(value))
            count = count - value
            print(str(count))
            
        elif key in positive_list:
            print("positive: " + str(count)+" + "+str(value))
            count = count + value
    
    return count

def campare(url, words):
    
    for word in words
    
    matplotlib.pyplot.bar(words,counts)



print(sentiment("texts/Tom Sawyer"))