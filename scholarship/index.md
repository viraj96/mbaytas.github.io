---
layout: default
title: scholarship
---

# scholarship

## solutions for Exercises in Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

### Chapter 1: Using neural nets to recognize handwritten digits
      
#### Sigmoid neurons simulating perceptrons, part I
      
(2) in the textbook gives the 'perceptron rule' as:

$ output = {(0 if w * x + b <= 0), (1 if w * x + b > 0):} $

Multiplying the weights and biases with a positive constant gives:

$ output = {(0 if cw * x + cb <= 0), (1 if cw * x + cb > 0):} $

Since the right side on both pieces of the equation is equal to zero, dividing both sides by $ c $ yields the original perceptron rule; hence, the behaviour of the network doesn't change when all weights and biases are multiplied with a positive constant.

#### Sigmoid neurons simulating perceptrons, part II 

Following (3) and (4) in the textbook, the output of a sigmoid neuron can be written as:

$ sigma(x) = 1 / (1 + e ^ (- w * x - b)) $

Multiplying all weights and biases by $ c > 0 $ as $ c -> oo $ gives us a step function:

$ lim_(c -> oo) 1 / (1 + e ^ (-c(w * x + b))) = {(0 if w * x + b < 0), (1 if w * x + b > 0):} $

This indeed fails to produce a binary classification when $ w * x + b = 0 $:

$ lim_(c -> oo) 1 / (1 + e ^ (-c(w * x - b))) = 1 / 2 $

#### determining the bitwise representation of a digit by adding an extra layer

We can begin figuring out the weights and biases for the new output layer by writing out what each digit will look like in 4-bit binary:

    0 -> 0 0 0 0
    1 -> 0 0 0 1
    2 -> 0 0 1 0
    3 -> 0 0 1 1
    4 -> 0 1 0 0
    5 -> 0 1 0 1
    6 -> 0 1 1 0
    7 -> 0 1 1 1
    8 -> 1 0 0 0
    9 -> 1 0 0 1

Following http://datascience.stackexchange.com/questions/6639/:

> Each output neuron should have a positive weight between itself and output neurons which should be on to represent it, and a negative weight between itself and output neurons that should be off. The values should combine to be large enough to cleanly switch on or off, so I would use largish weights, such as +10 and -10.
>
> If you have sigmoid activations here, the bias is not that relevant. You just want to simply saturate each neuron towards on or off.

Thus, one set of weights to the new output layer can be:

    w_(1k)^n = {-10,  10, -10,  10, -10,  10, -10,  10, -10,  10}
    w_(2k)^n = {-10, -10,  10,  10, -10, -10,  10,  10, -10, -10}
    w_(3k)^n = {-10, -10, -10, -10,  10,  10,  10,  10, -10, -10}
    w_(4k)^n = {-10, -10, -10, -10, -10, -10, -10, -10,  10,  10}
    
    
#### proof that gradient descent is the optimal strategy for minimizing a cost function

See:

- [http://math.stackexchange.com/questions/1688662/tricky-proof-of-a-result-of-michael-nielsens-book-neural-networks-and-deep-lea](http://math.stackexchange.com/questions/1688662/tricky-proof-of-a-result-of-michael-nielsens-book-neural-networks-and-deep-lea)
- [https://www.quora.com/How-does-one-prove-that-Gradient-Descent-is-the-optimal-strategy-for-minimizing-cost-function-using-Cauchy-Schwarz-inequality](https://www.quora.com/How-does-one-prove-that-Gradient-Descent-is-the-optimal-strategy-for-minimizing-cost-function-using-Cauchy-Schwarz-inequality) -> [http://www.princeton.edu/~amirali/Public/Teaching/ORF363_COS323/F14/ORF363_COS323_F14_Lec8.pdf](http://www.princeton.edu/~amirali/Public/Teaching/ORF363_COS323/F14/ORF363_COS323_F14_Lec8.pdf)

#### geometric interpretation of what gradient descent is doing in the one-dimensional case

#### online learning

#### activations vector in component form

$ a' = sigma(wa + b) $

$ = sigma([(w_(1,1), w_(1,2), ..., w_(1,k)), (w_(2,1), w_(2,2), ..., w_(2,k)), (..., ..., ..., ...), (w_(j,1), w_(j,2), ..., w_(j,k))] [(a_1), (a_2), (...), (a_k)] + b) $

$ = sigma(sum_j [w_(j,1), w_(j,2), ..., w_(j,k)] [(a_1), (a_2), (...), (a_k)] + b) $

$ = sigma(sum_j w_jx_j + b) $

$ = 1 / (1 + e ^ (-sum_j w_jx_j - b)) $

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    asciimath2jax: {
      delimiters: [ ['$','$'] ]
    },
  });
</script>
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=AM_HTMLorMML"></script>
