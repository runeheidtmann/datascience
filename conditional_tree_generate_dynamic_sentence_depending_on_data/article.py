# -----------------------------------------------------------
# This is a survey into the world of natural language generation.
# In the world of journalism, where we need to have non-black-box systems, rule based systems for NLG still rules.
# 
# 
# The following is a conditional tree, that can generate dynamic sentences.
#
#
# The tree will first asses the data-input, and it will then print out the corresponding sentences.
# 
# Example:
# Tree structure(sentence || condition): 
# 
#    |__Runes programming skills are  || {runes_skills}>100
#        |__through the roof with scores between 500 and 1000! || {runes_skills}>=500 and {runes_skills}<1000
#
# Tree output(runes_skills: 501)
# "Runes programming skills are through the roof with scores between 500 and 1000!""
#
# This is a very practical approach to NLG, because the flat json-database structure is very easy to build and maintain in a visual frontend.
# In this way you can build arbitrarily deep and complex trees, that write really varied and complex news articles from data.
#  
# -----------------------------------------------------------

import json
import os
class Article : 
    def __init__(self,data,condition) :
        self.data = data
        self.condition = condition
        self.children = []
        self.parent = None
    
    def add_child(self, child) :
        child.parent = self
        self.children.append(child)
    
    def get_level(self) :
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level
        
    def print_tree(self,env) :
        if self.condition:
            if eval(self.condition.format(**env)): 
                prefix = ' ' * self.get_level()*3
                prefix = prefix + '|__'
                print(prefix+self.data+ " || "+ self.condition)
            
                if self.children:
                        for child in self.children:
                            child.print_tree(env)
        else: 
            prefix = ' ' * self.get_level()*3
            prefix = prefix + '|__'
            print(prefix+self.data)
        
            if self.children:
                    for child in self.children:
                        child.print_tree(env)
    
    def print_article(self,env):

        if self.condition:
            if eval(self.condition.format(**env)): 
                print(self.data,end="")
            
                if self.children:
                        for child in self.children:
                            child.print_article(env)
        else: 
            print(self.data,end="")
            if self.children:
                    for child in self.children:
                        child.print_article(env)
    
def build_tree(js):
    for dic in js:
        root = Article(dic['text'],dic['condition'])

        if dic['children']:
            for child in dic['children']:
                root.add_child(build_tree([child]))
            
    return root



article_data = {'runes_skills': 501}


with open("tree_structure.json", "r") as tree:
    data = json.load(tree) 
    from_root = build_tree(data)
    from_root.print_tree(article_data)

    from_root.print_article(article_data)


    
    
  