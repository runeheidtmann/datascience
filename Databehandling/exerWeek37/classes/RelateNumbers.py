# -*- coding: utf-8 -*-
# Class that relates numbers to one another
# Implement in main class by:
# 
# converterProgram = RelateNumbers.RelateNumbers()
# converterProgram.run()
class RelateNumbers:
    
    def runNumberEval(self):
        print("Hi, this program will\nevaluate the relation between two numbers.")
        print("Please type in the first number")
        
        firstInputValidated = False
        secondInputValidated = False
        firstNum = ""
        secondNum = ""
        
        while not firstInputValidated:
            firstNum = input('> ')
            firstInputValidated = self.isItValid(firstNum)
            if firstInputValidated:
                firstNum = float(firstNum)  
            else:
                print("Only numeric inputs")
            
        while not secondInputValidated:
            secondNum = input('> ')
            secondInputValidated = self.isItValid(secondNum)
            if secondInputValidated:
                secondNum = float(secondNum)
            else:
                print("Only numeric inputs")            
        
        if firstNum > secondNum:
            print(str(firstNum)+" is bigger than "+str(secondNum))
        
        if firstNum < secondNum:
            print(str(secondNum)+" is bigger than "+str(firstNum))
        
        if firstNum is secondNum:
            print(str(firstNum)+" is equal to "+str(secondNum))
    
    def isItValid(self,num):
        try:
            okayfloat = float(num)
            return True
        
        except ValueError:
            return False