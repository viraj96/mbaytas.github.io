---
layout: default
title: scholarship
---

# scholarship

## solutions for Exercises in Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

### Chapter 1: Using neural nets to recognize handwritten digits</h3>
      
#### Sigmoid neurons simulating perceptrons, part I
      
(2) in the textbook gives the 'perceptron rule' as:

<p>
`output = {(0 if w * x + b <= 0), (1 if w * x + b > 0):}`
</p>

Multiplying the weights and biases with a positive constant gives:

<p>
`output = {(0 if cw * x + cb <= 0), (1 if cw * x + cb > 0):}`
</p>

Since the right side on both pieces of the equation is equal to zero, dividing both sides by `c` yields the original perceptron rule; hence, the behaviour of the network doesn't change when all weights and biases are multiplied with a positive constant.

#### Sigmoid neurons simulating perceptrons, part II 

Following (3) and (4) in the textbook, the output of a sigmoid neuron can be written as:

`sigma(x) = 1 / (1 + e ^ (- w * x - b))`

Multiplying all weights and biases by `c > 0`, as `c -> oo`:

`lim_(c -> oo) 1 / (1 + e ^ -c(w * x + b)) = {(0 if w * x + b < 0), (1 if w * x + b > 0):}`

This indeed fails when `w * x + b = 0`:

`lim_(c -> oo) 1 / (1 + e ^ -c(w * x - b)) = 1 / 2`

#### Determining the bitwise representation of a digit by adding an extra layer

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
