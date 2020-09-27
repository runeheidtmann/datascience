# -*- coding: utf-8 -*-

class EvenDivision:
    
    def howEvenDivides(self,n,num):
        
        count = 0
        
        for i in range(2,n,2):
            if i % num == 0:
                count = count + 1
                print(str(count)+". number that divides with "+str(num)+" is: "+str(i))
        
        print("Total numbers that divide with your input: "+str(count))
                