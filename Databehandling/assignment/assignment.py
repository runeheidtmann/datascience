# -*- coding: utf-8 -*-
# =============================================================================
# Assignment by Rune Heidtmann
# =============================================================================

import math  
import os
import re
import matplotlib.pyplot as plt


def pythagorean():

# =============================================================================
# Method Pythagorean.
# 
# Asks user to input side of a triangle.
# If possible the method calculates the length of the missing side.
# If possible the method plots the triangle.
# Else the method prints an error-message.
# 
# =============================================================================
    
    print("Pythagorean theorem states that a^2 + b^2 = c^2")
    print("Put in the sides you know. Put in ?-mark, if the side is unknown: ")
    
    result = None
    pyth_form = None
    triangle = {"a":None,"b":None,"c":None}
    
    for side in triangle:
        inputs = input("Length of side "+side+": ")
        if inputs == '?':
            pyth_form = side
            triangle[side] = inputs
        else:
            triangle[side] = int(inputs)
            
    
   # Storing triangle side in variables for improved readabilty
    a = triangle["a"]
    b = triangle["b"]
    c = triangle["c"]
   
    if pyth_form == "a":
        if c*c - b*b > 0:
            result = math.sqrt(c*c-b*b)
            a = result
    
    elif pyth_form == "b":
        if c*c - a*a > 0:
            result = math.sqrt(c*c-a*a)
            b = result
    
    elif pyth_form == "c":
            result = math.sqrt(a*a+b*b)
            c = result
    
    if result != None:
        visualize_triangle(a,b,c)

    # If there is a result, print it, else tell user rectangle is not right
    print("The side " + pyth_form + " = " + str(result)) if result != None \
    else print("Impossible to draw a right triangle with inputted side lengths.")



def median(list_of_numbers):
# ========================================================================
# A method that takes a list of objects L and
# returns a sub-list that contains those items that occur only once in L.
# ========================================================================
    sorted_list = sorted(list_of_numbers)
    
    # Find middle index by floor division.
    median_index = len(sorted_list) // 2
    
    if len(sorted_list)%2 != 0:
        #Odd number sets median:
        return sorted_list[median_index]
    
    else:
        #even number sets median is to average of the to middle elements.
        return (sorted_list[median_index]+sorted_list[median_index-1]) / 2 
        
def unique(list_of_objects):
# =============================================================================
#   A Method that takes a list and returns a subset, of the unique elements
#   in the list.
# =============================================================================
    count_objects_dict = {}
    for element in list_of_objects:
        if element not in count_objects_dict:
            count_objects_dict[element] = 1
        else:
            count_objects_dict[element] += 1
    
    uniquelist = []
    for item,value in count_objects_dict.items():
        if value == 1:
            uniquelist.append(item)
    
    return uniquelist
    
def characters(textfile):
# =============================================================================
# Method that takes a loaded textfile.
# Loops through the text and counts the frequency of each character.
# Lastly et prints the frequency-dictionary in a formatted way.
# =============================================================================
    frequency = {}
    text = textfile.read()
    for char in text:
        if char not in frequency:
            frequency[char] = 1
        else:
            frequency[char] += 1
    
    # Call homemade sort method.
    sorted_dict = sort_dict(frequency)
    
    # print stats:
    for key,value in sorted_dict.items():
        print("'"+key+"'"+": "+str(value))

def sort_dict(a_dict):
    # Make a sorted dict. First sort keys. Thereafter make new sorted dict.
    list_of_sorted_keys = sorted(a_dict.keys())
    sorted_dict = {}
    
    for key in list_of_sorted_keys:
        sorted_dict[key] = a_dict[key]
    
    return sorted_dict

def count(term):
    #load list of chapterfiles
    frequency_dict = {}
    path = "genesis/"
    chapterlist = os.listdir(path)
    
    #Sorting for "chapter-xx" type naming.
    chapters = realsort(chapterlist)         

    chapter_count = 0;
    for chapter in chapters:
        f = open(path+chapter, "r")
        text = f.read().lower()
        words = text.split()
        
        term_count = 0
        for i in range(0,len(words)):
            words[i] = re.sub(r"[^a-z A-Z0-9 ]","",words[i])
        for word in words:
            if word == term.lower():
                term_count += 1
        
        chapter_count += 1
        frequency_dict[chapter_count] = term_count
    
    visualize_dict(frequency_dict,"Your term count","Chapter", "Term Count")
    

def visualize_dict(D,title,xlabel,ylabel):     
# =============================================================================
#  Method for visualizing a dictionary.   
# =============================================================================
    #Set figure size
    width = 16
    height = 9
    width_height = (width, height)
    plt.figure(figsize=width_height)
    
    #create plot and set labels and title
    plt.plot(range(len(D)), list(D.values()))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    plt.show()

def visualize_triangle(len_a, len_b, len_c):
# =============================================================================
#  Let's say a triangles right corner of ab is in point (0,0).
#     Then: 
#     vector a = [0 , len_a]
#     vector b = [len_b , 0]
#     vector c = a + b
# =============================================================================
    
    #The fourth set of coordinates(0,0) is to let the tiangle-drawing close up
    vector_x_coordinates = [0,0,len_b,0]
    vector_y_coordinates = [0,len_a,0,0]
    
    plt.plot(vector_x_coordinates, vector_y_coordinates)
    plt.show()

def realsort(textlist):
# =============================================================================
#     The sorted() algorithm sorts a list of string lexiographically. This
#     means that it evaluates chapter-2 higher than chapter-11. This is
#     unwanted. So vi need this workaround sorting method.
#       
#   Takes a list, splits the list up into lists of the same size string.
#   Since: len(chapter-123) > len(chapter-14) > len(chapter-2).
#   Then it sorts the lists, and merges them.
#   
# =============================================================================
    split_lists = {}
    
    for text in textlist:
        if len(text) not in split_lists:
            split_lists[len(text)] = [text]
        else:
            split_lists[len(text)].append(text)
            
    #Sort the dict
    sorted_dict = sort_dict(split_lists)
    
    #merge lists
    result_list = []
    for key,value in sorted_dict.items():
        for item in value:
            result_list.append(item)
    
    return result_list
        
def multiply_matrices(a,b):
    
    size_a = matrixSize(a)
    size_b = matrixSize(b)
    
    is_defined = is_product_defined(size_a,size_b)
    
    if is_defined:
        return get_matrix_product(a,b)
    
    return None

def get_matrix_product(a,b):
    
    #init multidimensional array with right matrix size.
    product = [[0] * len(b[0]) for i in range(len(a))]
    
    for i in range(len(product)):
        for j in range(len(product[0])):
            for n in range(len(b)):
                product[i][j] += a[i][n]*b[n][j]
            
    return product
        

def is_product_defined(a,b):
   
   if a and b is not None:    
       return a[1] == b[0]
   return False

def matrixSize(matrix):
    
    rows = None
    coloumns = None
    len_of_rows = len(matrix[0])
    same_size = True
    
    for row in matrix:
        if len(row) != len_of_rows:
            same_size = False
    
    if same_size:
        coloumns = len(matrix[0])
        rows = len(matrix)
        
        return [rows,coloumns]
    else:
        return None
    

print(multiply_matrices([[1,2],
                         [2,3],
                         [3,4],
                         ],
                        
                        [[9,8,7],
                         [9,1,4]
                         ]))


######## Test area uncomment for testing #############    

#pythagorean()

#list_of_numbers = [1,2,3,4,7,12,54,645,234,12,434,6]
#print(median(list_of_numbers))

#list_of_objects = ["ew","unique","ew","efw","efw"]
#print(unique(list_of_objects))

#f = open('raven.txt','r')
#characters(f)

#count('God')


    
















