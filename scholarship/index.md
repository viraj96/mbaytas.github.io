---
layout: default
title: scholarship
---

# scholarship

## solutions for Exercises in Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

### chapter 1: Using neural nets to recognize handwritten digits
      
#### exercise: Sigmoid neurons simulating perceptrons, part I
      
(2) in the textbook gives the 'perceptron rule' as:

$ output = {(0 if w * x + b <= 0), (1 if w * x + b > 0):} $

Multiplying the weights and biases with a positive constant gives:

$ output = {(0 if cw * x + cb <= 0), (1 if cw * x + cb > 0):} $

Since the right side on both pieces of the equation is equal to zero, dividing both sides by $ c $ yields the original perceptron rule; hence, the behaviour of the network doesn't change when all weights and biases are multiplied with a positive constant.

#### exercise: Sigmoid neurons simulating perceptrons, part II 

Following (3) and (4) in the textbook, the output of a sigmoid neuron can be written as:

$ sigma(x) = 1 / (1 + e ^ (- w * x - b)) $

Multiplying all weights and biases by $ c > 0 $ as $ c -> oo $ gives us a step function:

$ lim_(c -> oo) 1 / (1 + e ^ (-c(w * x + b))) = {(0 if w * x + b < 0), (1 if w * x + b > 0):} $

This indeed fails to produce a binary classification when $ w * x + b = 0 $:

$ lim_(c -> oo) 1 / (1 + e ^ (-c(0))) = 1 / 2 $

#### exercise: determining the bitwise representation of a digit by adding an extra layer

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

Following [http://datascience.stackexchange.com/questions/6639/](http://datascience.stackexchange.com/questions/6639/):

> Each output neuron should have a positive weight between itself and output neurons which should be on to represent it, and a negative weight between itself and output neurons that should be off. The values should combine to be large enough to cleanly switch on or off, so I would use largish weights, such as +10 and -10.

> If you have sigmoid activations here, the bias is not that relevant. You just want to simply saturate each neuron towards on or off.

Thus, one set of weights to the new output layer can be:

    w_(1k)^n = {-10,  10, -10,  10, -10,  10, -10,  10, -10,  10}
    w_(2k)^n = {-10, -10,  10,  10, -10, -10,  10,  10, -10, -10}
    w_(3k)^n = {-10, -10, -10, -10,  10,  10,  10,  10, -10, -10}
    w_(4k)^n = {-10, -10, -10, -10, -10, -10, -10, -10,  10,  10}
    
    
#### exercise: proof that gradient descent is the optimal strategy for minimizing a cost function

See:

- [http://math.stackexchange.com/questions/1688662/tricky-proof-of-a-result-of-michael-nielsens-book-neural-networks-and-deep-lea](http://math.stackexchange.com/questions/1688662/tricky-proof-of-a-result-of-michael-nielsens-book-neural-networks-and-deep-lea)
- [https://www.quora.com/How-does-one-prove-that-Gradient-Descent-is-the-optimal-strategy-for-minimizing-cost-function-using-Cauchy-Schwarz-inequality](https://www.quora.com/How-does-one-prove-that-Gradient-Descent-is-the-optimal-strategy-for-minimizing-cost-function-using-Cauchy-Schwarz-inequality) -> [http://www.princeton.edu/~amirali/Public/Teaching/ORF363_COS323/F14/ORF363_COS323_F14_Lec8.pdf](http://www.princeton.edu/~amirali/Public/Teaching/ORF363_COS323/F14/ORF363_COS323_F14_Lec8.pdf)

#### exercise: geometric interpretation of what gradient descent is doing in the one-dimensional case

When CC is a function of just one variable, the change $ Delta C $ in $ C $ produced by a small change $ Delta x $ in $ x $ is:

$ Delta C = (dC)/(dx) Delta x $

Choosing $ Delta x = - eta (dC)/(dx) $:

$ Delta C = -eta ((dC)/(dx))^2 $
 
#### exercise: online learning

See:

- <https://www.quora.com/What-are-the-pros-and-cons-of-offline-vs-online-learning]>
- <http://stats.stackexchange.com/questions/70761/what-is-the-difference-between-online-and-batch-learning>
- <http://stats.stackexchange.com/questions/897/online-vs-offline-learning>
- <http://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent>

#### exercise: activations vector in component form

$ a' = sigma(wa + b) $

$ = sigma([(w_(1,1), w_(1,2), ..., w_(1,k)), (w_(2,1), w_(2,2), ..., w_(2,k)), (..., ..., ..., ...), (w_(j,1), w_(j,2), ..., w_(j,k))] [(a_1), (a_2), (...), (a_k)] + b) $

$ = sigma(sum_j [[w_(j,1), w_(j,2), ..., w_(j,k)]] [(a_1), (a_2), (...), (a_k)] + b) $

$ = sigma(sum_j w_jx_j + b) $

$ = 1 / (1 + e ^ (-sum_j w_jx_j - b)) $

#### exercise: accuracy of network with only 2 layers

    >>> net = network.Network([784, 10])
    >>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    
    Epoch 0 (0:00:03.147000): 7519 / 10000
    Epoch 1 (0:00:03.092000): 8306 / 10000
    Epoch 2 (0:00:03.144000): 8302 / 10000
    Epoch 3 (0:00:03.157000): 8337 / 10000
    Epoch 4 (0:00:03.124000): 8350 / 10000
    Epoch 5 (0:00:03.148000): 8377 / 10000
    Epoch 6 (0:00:03.135000): 8363 / 10000
    Epoch 7 (0:00:03.127000): 8366 / 10000
    Epoch 8 (0:00:03.157000): 8369 / 10000
    Epoch 9 (0:00:03.162000): 8360 / 10000
    Epoch 10 (0:00:03.127000): 8377 / 10000
    Epoch 11 (0:00:03.142000): 8371 / 10000
    Epoch 12 (0:00:03.116000): 8358 / 10000
    Epoch 13 (0:00:03.127000): 8376 / 10000
    Epoch 14 (0:00:03.135000): 8348 / 10000
    Epoch 15 (0:00:03.165000): 8380 / 10000
    Epoch 16 (0:00:03.214000): 8376 / 10000
    Epoch 17 (0:00:03.160000): 8382 / 10000
    Epoch 18 (0:00:03.194000): 8373 / 10000
    Epoch 19 (0:00:03.347000): 8329 / 10000
    Epoch 20 (0:00:03.187000): 8372 / 10000
    Epoch 21 (0:00:03.198000): 8374 / 10000
    Epoch 22 (0:00:03.180000): 8366 / 10000
    Epoch 23 (0:00:03.121000): 8376 / 10000
    Epoch 24 (0:00:03.109000): 8361 / 10000
    Epoch 25 (0:00:03.125000): 8391 / 10000
    Epoch 26 (0:00:03.089000): 8356 / 10000
    Epoch 27 (0:00:03.117000): 8371 / 10000
    Epoch 28 (0:00:03.132000): 8359 / 10000
    Epoch 29 (0:00:03.123000): 8370 / 10000

### chapter 2: How the backpropagation algorithm works

#### problem: Alternate presentation of the equations of backpropagation

##### (1)

Supposing there are $ N $ nodes in the output layer:

$ delta_j^L = (del C) / (del a_j^L) sigma′(z_j^L) $

$ delta^L = grad_a C o. sigma′(z^L) $

$ = [ [(del C) / (del a_1^L)], [(del C) / (del a_2^L)], [(del C) / (del a_3^L)], [...], [(del C) / (del a_N^L)] ] o. [ [sigma′(z_1^L)], [sigma′(z_2^L)], [sigma′(z_3^L)], [...], [sigma′(z_N^L)] ] $

$ = [ [sigma′(z_1^L), 0, 0, ..., 0], [0, sigma′(z_2^L), 0, ..., 0], [0, 0, sigma′(z_3^L), ..., 0], [..., ..., ..., ..., ...], [0, 0, 0, ..., sigma′(z_N^L)] ] [ [(del C) / (del a_1^L)], [(del C) / (del a_2^L)], [(del C) / (del a_3^L)], [...], [(del C) / (del a_N^L)] ] $

$ = Sigma′(z^L) grad_a C $

##### (2)

Supposing there are $ k $ nodes in layer $ l $ and $ j $ nodes in layer $ l + 1 $:

$ delta^l = ((w^(l+1))^T delta^(l+1)) o. sigma′(z^l) $

$ = [(w_(1,1)^(l+1), w_(1,2)^(l+1), ..., w_(1,k)^(l+1)), (w_(2,1)^(l+1), w_(2,2)^(l+1), ..., w_(2,k)^(l+1)), (..., ..., ..., ...), (w_(j,1)^(l+1), w_(j,2)^(l+1), ..., w_(j,k)^(l+1))]^T ... $

#### exercise: proofs

#### exercise: Backpropagation with a single modified neuron

#### exercise: Backpropagation with linear neurons

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  asciimath2jax: {
    delimiters: [ ['$','$'] ]
  },
});
</script>
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=AM_HTMLorMML"></script>

<script>
$( document ).ready(function() {
  $('h4').each(function(){
    var n = $(this).next(); 

    $(this).replaceWith("<div class='panel-heading panelall'><h4 class='panel-title'>" + $(this).text() + "</h4></div>");

    while (n.is('p, pre, blockquote, ul, ol, h5, div.highlighter-rouge')) {
      $(n).addClass("panelall panelbody");
      n = $(n).next();
    }

    $(".panelall").wrapAll("<div class='panel panel-default'></div>");
    $(".panelbody").wrapAll("<div class='panel-body'></div>");

    $(".panelall").removeClass("panelall");
    $(".panelbody").removeClass("panelbody");
  });
});
</script>
