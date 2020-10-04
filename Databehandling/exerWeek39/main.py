import json
file = open('dbl.json')
dbl = json.load(file)


def born_in(nameList,city):
    newlist = []
    
    for person in nameList:
        if person['p'] == city:
            newlist.append(person['name'])
    
    return newlist

def print_gender():
    males = 0
    females = 0
    
    for person in dbl:
        if person['g'] == 'm':
            males += 1
        else:
            females += 1

    print(str(males) +" "+ str(round(males/len(dbl)*100,2))+"% / " +str(females)+" "+str(round(females/len(dbl)*100,2))+"%")

def aristocrats(persons_list):
    
    titles = [" af "," von "," van "," de "]
    aristocrats = []
    
    for person in persons_list:
        for title in titles:
            if person['name'].find(title) > 0:
                aristocrats.append(person)
    
    return aristocrats

def first_name(persons_list,name):
    
    result_list = []
    
    for person in persons_list:
        names = person['name'].split() 
        if names[0] == name:
            result_list.append(person)
    
    return result_list
    
def born_rural(persons_list):
    
    cities = ['Copenhagen','Aarhus','Odense']
    rural_borns = []
    
    for person in persons_list:       
        
        rural = True
        
        for city in cities:
            if person['p'] == city:
                rural = False
        
        if rural:
            rural_borns.append(person)
                
    return rural_borns

def older_than(persons_list,age):
    
    result_list = []
    
    for person in persons_list:
        
        years = person['l'].split("-")
        
        if len(years) == 2 and years[0] != '' and years[1] != '':
            if -eval(person['l']) > age:
               result_list.append(person)
    
    return result_list            
        
def death_between(persons_list,min_year,max_year):
    
    result_list = []
    
    for person in persons_list:
        years = person['l'].split("-")
        if len(years) == 2 and years[0] != '' and years[1] != '':
            if min_year < eval(years[1]) < max_year:
                result_list.append(person)
                
    return result_list
            
def painters(persons_list):
    
    result_list =[]
    
    for person in persons_list:
        if "painter" in person['o'] and 1800 < person['y'] < 1850:
            result_list.append(person)
    return result_list

def occupations(persons_list,o_list):
       
    result_list = []
    
    for person in persons_list:
        all_o_is_in = True
        
        for o in o_list:
            if o not in person['o']:
                all_o_is_in = False
        
        if all_o_is_in:
            result_list.append(person)
    
    return result_list


def lonName(persons):
    longest = "";
    for person in persons:
        if len(person['name']) > len(longest):
            longest = person['name']
    
    return longest
        
    
#Testkoden:

print(lonName(dbl))