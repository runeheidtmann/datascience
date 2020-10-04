# This is a question-based-game, that we developed in class.
# It did this by my self.

import random
f = open("questions.txt","r+")

questions = [{
    "question": "Which of these is not one of the three branches of the US government?",
    "answers": ["Judicial","Executive","Parliamentary","Legislative"],
    "correct": 3
}]

def userQuestion():
    pass
    
def addQuestion(question, answers, correct):
    new_question = {"question":question,"answers":answers,"correct":correct}
    
    questions.append(new_question)
    
def storeQuestion(question, answers, correct):
    pass
    
def loadQuestions(file):
    all_questions = file.read()
    questions_as_list = all_questions.splitlines()
    
    #Split questions up, and add them as questions in propper forrmat in program
    
    for q in questions_as_list:
        
        qlist = q.split("\t")
        
        current_q = qlist[0]
        current_q_answers = [qlist[1],qlist[2],qlist[3],qlist[4]]
        current_right_answer = int(qlist[5])
        
        addQuestion(current_q, current_q_answers, current_right_answer)        
    
def printQuestion(q):
    
    print(q["question"])
    
    counter = 1
    for a in q["answers"]:
        print(str(counter)+": "+a)
        counter += 1
    
    
def getRandomQuestions(n):
    
    n_questions = []
    
    while len(n_questions) < n:
        rand_num = random.randint(0, len(questions))
        if questions[rand_num] not in n_questions:
            n_questions.append(questions[rand_num])
            
    return n_questions
        
    
def printMessage(score,n):
    print(f'You answered {score} out of {n}  questions correctly!!')
    
    percentage = score/n
    
    if percentage < 0.2:
        print('That was a poor performance')
    elif  0.2 < percentage < 0.4:
        print("That was a fair performance!")
    elif 0.4 < percentage < 0.6:
        print("Your performance was okay!")
    elif 0.6 < percentage < 0.8:
        print("Good performance!!")
    elif percentage > 0.8:
        print("Excellent performance!!")
    
def play(n):    
    loadQuestions(f)
    game_questions = getRandomQuestions(n)
    score = 0
    
    for q in game_questions:
        printQuestion(q)
        answer = int(input("Answer number: "))
        if answer-1 == q["correct"]:
            score += 1
    
    printMessage(score,n)
    

play(2)
