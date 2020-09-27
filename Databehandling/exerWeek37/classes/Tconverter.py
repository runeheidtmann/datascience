# -*- coding: utf-8 -*-
# Class that runs a program that converts degrees from F->C and C->
# Implement in main class by:
# 
# converterProgram = Tconverter.Tconverter
# converterProgram.run()

class Tconverter:
    
    def run():
        
        running = True
        
        while running:
            
            print("Type in temperature in celcius or fahrenheit - or quits")
            inTemp = input('> ')
            if inTemp == "quits":
                break
            
            if not all(inTemp.split("F")):
                # meas that inTemp found F and split the array
                # So temp was in Fahrenheit, go ahead and convert to celsius
                temp_arr = inTemp.split("F")
                temp_f = float(temp_arr[0])
                temp_c = round(((temp_f - 32)/1.8),2)
                print(inTemp + " = " + str(temp_c)+"C")
               
            if not all(inTemp.split("C")):
                # meas that inTemp found C and split the array
                # So temp was in C, go ahead and convert to F
                temp_arr = inTemp.split("C")
                temp_c = float(temp_arr[0])
                temp_f = round((temp_c*1.8+32),2)
                print(inTemp + " = " + str(temp_f)+"F")
                
            else:
                print("Wrong kind of input")