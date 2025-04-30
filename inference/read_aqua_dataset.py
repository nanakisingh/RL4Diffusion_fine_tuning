'''

    Helper function for test_inference_run.py

'''

import json

def read_aqua_data(json_file_path):
    # json_file_path = "dev.json"
    f = open(json_file_path, "r")

    questions = []
    answers = []

    for line in f: 
        # line = string. use json.loads(s) to convert to actual datatype 
        dict_format = json.loads(line)
        
        # Reformat question -> question + options 
        q_org = dict_format['question']
        options = " The multiple choice options are " + " ".join(dict_format['options'])
        q_new = q_org + options
        
        # Answer = rationale - has 'Correct Answer - ?'
        new_answer = dict_format['rationale'].replace('\n', '. ')
        new_answer = new_answer.replace('.. ', '. ')
        
        questions.append(q_new)
        answers.append(new_answer) 

    # Return questions and answers 
    return questions, answers