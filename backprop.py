import math
import random
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.header("_Simple 2 - 2 - 2 Neural Network_",divider="grey" )
st.write("Please input the number between 0 and 1")


print("Program started")
#input
x1 = 0.056
x2 = 0.027

#weights
w1 = random.random()
w2 = random.random()
w3 = random.random()

w4 = random.random()
w5 = random.random()
w6 = random.random()
w7 = random.random()
w8 = random.random()

print("Weights: ",w1,w2,w3,w4,w5,w6,w7,w8)

#bias
b1 = 0.35
b2 = 0.60

#output as input also known as target
ro1 = st.number_input("Enter your first desired output")
ro2 = st.number_input("Enter your second desired output")

#learning rate
l = 0.05

epoch = 0

def activation(in_):
    b = -(in_)
    a = 1/ (1 + math.exp(b))
    return a

def h_layer1(x1,x2,w1,w2,b1):
    h1in = x1*w1 + x2*w2 + b1
    h1out = activation(h1in)
    return h1out

def h_layer2(x1,x2,w3,w4,b1):
    h2in = x1*w3 + x2*w4 + b1
    h2out = activation(h2in)
    return h2out

loss = []

while epoch < 50001:
    





    h1in = x1*w1 + x2*w2 + b1
    h1out = activation(h1in)

    h2in = x1*w3 + x2*w4 + b1
    h2out = activation(h2in)

    h1 = h_layer1(x1,x2,w1,w2,b1)
    h2 = h_layer2(x1,x2,w3,w4,b1)
    o1in = h1*w5 + h2*w6 +b2
    o1out = activation(o1in)
    
    o2in = h1*w7 + h2*w8 +b2
    o2out = activation(o2in)

    e1 = ro1 - o1out
    e2 = ro2 - o2out
    et = 1/2 * ((e1**2) + (e2**2))
    loss.append(et)

    for i in range(5,9):
        if i == 5:
                in_w = h1out
                et_out = o1out - ro1
                out_in = o1out*(1-o1out)
                et_w = et_out * out_in * in_w
                w5 = w5 - l*et_w
        elif i == 6:
                in_w = h2out
                et_out = o1out - ro1
                out_in = o1out*(1-o1out)
                et_w = et_out * out_in * in_w
                w6 = w6 - l*et_w
        elif i == 7:
                in_w = h1out
                et_out = o2out - ro2
                out_in = o2out*(1-o2out)
                et_w = et_out * out_in * in_w
                w7 = w7 - l*et_w
        elif i == 8:
                in_w = h2out
                et_out = o2out - ro2
                out_in = o2out*(1-o2out)
                et_w = et_out * out_in * in_w
                w8 = w8 - l*et_w
    
    for x in range(1,5):
        if x == 1:
            in_w2 = x1
            out_in2 = h1out*(1- h1out)
            e1_in = (o1out - ro1) * o1out*(1 - o1out)
            e2_in = (o2out - ro2) * o2out*(1 - o2out)
            e1_out = e1_in * w5
            e2_out = e2_in * w7
            et_out = e1_out + e2_out
            et_w = in_w2 * out_in2 * et_out
            w1 = w1 - l*et_w
        elif x == 2:
            in_w2 = x2
            out_in2 = h1out*(1- h1out)
            e1_in = (o1out - ro1) * o1out*(1 - o1out)
            e2_in = (o2out - ro2) * o2out*(1 - o2out)
            e1_out = e1_in * w5
            e2_out = e2_in * w7
            et_out = e1_out + e2_out
            et_w = in_w2 * out_in2 * et_out
            w2 = w2 - l*et_w
        elif x == 3:
            in_w2 = x1
            out_in2 = h2out*(1- h2out)
            e1_in = (o1out - ro1) * o1out*(1 - o1out)
            e2_in = (o2out - ro2) * o2out*(1 - o2out)
            e1_out = e1_in * w6
            e2_out = e2_in * w8
            et_out = e1_out + e2_out
            et_w = in_w2 * out_in2 * et_out
            w3 = w3 - l*et_w
        elif x == 4:
            in_w2 = x2
            out_in2 = h2out*(1- h2out)
            e1_in = (o1out - ro1) * o1out*(1 - o1out)
            e2_in = (o2out - ro2) * o2out*(1 - o2out)
            e1_out = e1_in * w6
            e2_out = e2_in * w8
            et_out = e1_out + e2_out
            et_w = in_w2 * out_in2 * et_out
            w4 = w4 - l*et_w

    epoch = epoch + 1


st.write(f"Prediction 1: {o1out}")
st.write(f"Prediction 2: {o2out}")

st.header("_Loss curve_", divider=True)

df = pd.DataFrame(loss[::100])
st.line_chart(df)

print("Target: ",ro1)
print("Pred: ",o1out)
print("Target:",ro2)
print("Pred: ",o2out)

loss.clear()
