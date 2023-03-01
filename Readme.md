**README:**

In order to use the functions mentioned in this notebook you need to execute the notebook once in your system and then the function calls can be made as:

>* Forward Propogation: 
>>* ReLU : forw_relu(x)
>>>* returns y (shape same as input x)
>>* maxpool: forw_maxpool(x)
>>>* returns y (shape as x.shape//2)
>>* meanpool: forw_meanpool(x)
>>>* returns y (shape as x.shape//2)
>>* fully connect: forw_fc(x,w,b)
>>>* where w = weight and b = a scalar bias value
>>>* returns y, a scalar value
>>* softmax: forw_softmax(x)
>>>* returns y (shape same as input x)

>* Backward Propogation: 
>>* ReLU : back_relu(x,y,dzdy)
>>>* returns dzdx (shape same as input x) with values as dz/dx values
>>* maxpool: back_maxpool(x,y,dzdy)
>>>* returns dzdx (shape same as input x) with values as dz/dx values
>>* meanpool: back_meanpool(x,y,dzdy)
>>>* returns dzdx (shape same as input x) with values as dz/dx values
>>* fully connect: back_fc(x,w,b,y,dzdy)
>>>* returns dzdx (shape same as input x) with values as dz/dx values
>>* softmax: back_softmax (x,y,dzdy))
>>>* returns a list of derivatives on loss function with respect to the input (x), weights (w), and bias (b) as [dzdx,dzdw,dzdb]


 *(where **x** = input mxn matrix; 
 **y** in backward propogation are the ground truth labels from the forward pass of **x** on that particular function; 
 **dzdy** is the derivation of y with respect to the loss function - values given)*









