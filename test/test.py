from forw_maxpool import forw_maxpool
from back_maxpool import back_maxpool
from back_meanpool import back_meanpool
from forw_meanpool import forw_meanpool
from forw_relu import forw_relu
from back_relu import back_relu
from forw_softmax import forw_softmax
from back_softmax import back_softmax
from forw_fc import forw_fc
from back_fc import back_fc

import numpy as np
f = open("hw3testfile.txt", "r")
lst = f.read().split("\n")
def one_inp_to_correct_format(inp_lst):
    y= np.array(inp_lst[2:],dtype = np.float32).reshape(int(inp_lst[0]),int(inp_lst[1]), order= 'F')
    return y

def lst_inp_to_func_inp(inputs):
    inputs_to_func = []
    for inp in inputs:
        inputs_to_func.append(one_inp_to_correct_format(inp.split()))
    return tuple(inputs_to_func)

def handle_input_output(cmd_num):
    num_of_input = int(lst[cmd_num])
    cmd_num += 1
    func_inputs = lst_inp_to_func_inp(lst[cmd_num:cmd_num+num_of_input])
    cmd_num += num_of_input
    return cmd_num,func_inputs

def _check(func_output, test_output):
    ## garantued one vector only
    m,n = func_output.shape
    if (m,n)!=test_output.shape:
        print(f"Wrong Shapes please check, Expected {test_output.shape} but got {test_output.shape}")
    
    diff = abs(np.round(func_output,4)-test_output)
    return diff<1e-3

def check_equality(func_output, test_output):
    if len(test_output)==1:
        return _check(func_output,test_output[0])
    
    ## multiple outputs:
    diffs = []
    for i in range(len(test_output)):
        diffs.append(_check(func_output[i],test_output[i]))
    return diffs

def _check_array(a):
    if type(a)!=list:
        return a.all()
    ans = True
    for i in a:
        ans = ans & i.all()
    return ans
    
wrong = 0
cmd_num = 0
while(cmd_num<len(lst)-1):
    assert lst[cmd_num][0].isalpha()
    func_name = lst[cmd_num]
    cmd_num += 1
    
    ## input to fuction
    cmd_num,func_inputs = handle_input_output(cmd_num)
    
    ## actual output
    cmd_num,test_output = handle_input_output(cmd_num)
    
    ##
    func = globals()[func_name]
    func_output = np.array(func(*func_inputs))
    

    a = check_equality(func_output,test_output)
    if _check_array(a)==False:
        print("==================")
        print("Not Working "+func_name)
        wrong += 1
        print("Func o/p: ",func_output)
        print("Actual: ",test_output)
        print("difference:")
        print(a)
        print("==================")
    else:
        print("Working "+func_name)

print(f"# Wrong ans: {wrong}")
