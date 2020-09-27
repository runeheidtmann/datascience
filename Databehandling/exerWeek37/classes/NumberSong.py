# -*- coding: utf-8 -*-
class NumberSong:
    
    def __init__(self,n):
        self.n = n
        self.singMeASong()
    
    def singMeASong(self):
        
        for i in range(0,self.n):
            if i < self.n-1:
                print(str(self.n-i)+" books on Python on the shelf. Take one down, pass it around, "+str(self.n-i-1)+" books left.")
            else:
                print(str(self.n-i)+" books on Python on the shelf. Take one down, pass it around, no more books left.")
            