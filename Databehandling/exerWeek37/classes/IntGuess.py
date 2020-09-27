# -*- coding: utf-8 -*-
import random

class IntGuess:
   
    def __init__(self):
        self.secret = random.randint(1,1001)
        
    def runGame(self):
        
        gameActice = False
        print("Hi, do you want to play a fun game?")
        inputs = input('Y or N > ')
        
        if(inputs == "N"):
            print("Your loss, have a good day.")
        elif(inputs == "Y"): 
            
            gameActive = True
            print("Cool, i think of a number larger than 0 and less than 1001")
            print("Guess it")
            
            while gameActive:
                try:
                    guess = int(input('> '))
               
                    if(guess == self.secret):
                        gameActive = False
                        print('You guessed it!!!')
                    
                    elif(guess > self.secret):
                        print("Nope, my secret number is LOWER than that")
                    elif(guess < self.secret):                
                        print("Nope, my secret number is LARGER than that")
               
                except ValueError:
                    print("Stupid, guess a NUMBER!")
                
            print("The Game is over")