# Neural Network classification with PyTorch
- --> Classification is a problem of predicting whether something is one thing or another (there can be multiple things as the options).

# 1. Make classification data and get it ready

# 1.1 Check input and output shapes

# 1.2 Turn data into tensors and create train and test splits

# 2. Building a model
- Setup device agonistic code so our code will run on an accelerator (GPU) if there is one
- Construct a model (by subclassing nn.Module)
- Define a loss function and optimizer
- Create a training and test loop

# 2.1 Setup loss function and optimizer 
Which loss function or optimizer should you use?

Again... this is problem specific.

For example for regression you might want MAE or MSE (mean absolute error or mean squared error).

For classification you might want binary cross entropy or categorical cross entropy (cross entropy).

As a reminder, the loss function measures how wrong your models predictions are.

And for optimizers, two of the most common and useful are SGD and Adam, however PyTorch has many built-in options.

- For some common choices of loss functions and optimizers - https://www.learnpytorch.io/02_pytorch_classification/#21-setup-loss-function-and-optimizer
- For the loss function we're going to use torch.nn.BECWithLogitsLoss(), for more on what binary cross entropy (BCE) is, check out this article - https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
- For a defintion on what a logit is in deep learning - https://stackoverflow.com/a/52111173/7900723
- For different optimizers see torch.optim

# 3. Train model
--> To train our model, we're going to need to build a training loop with the following steps:
- Forward pass
- Calculate the loss
- Optimizer zero grad
- Loss backward (backpropagation)
- Optimizer step (gradient descent)

# 3.1 Going from raw logits -> prediction probabilities -> prediction labels

Our model outputs are going to be raw logits.

We can convert these logits into prediction probabilities by passing them to some kind of activation function (e.g. sigmoid for binary classification and softmax for multiclass classification).

Then we can convert our model's prediction probabilities to prediction labels by either rounding them or taking the argmax().

# 3.2 Building a training and testing loop

# 4. Make predictions and evaluate the model

From the metrics it looks like our model isn't learning anything...

So to inspect it let's make some predictions and make them visual!

In other words, "Visualize, visualize, visualize!"

To do so, we're going to import a function called plot_decision_boundary() - https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py

# 5. Improving a model (from a model perspective)

- Add more layers - give the model more chances to learn about patterns in the data
- Add more hidden units - go from 5 hidden units to 10 hidden units
- Fit for longer
- Changing the activation functions
- Change the learning rate
- Change the loss function

These options are all from a model's perspective because they deal directly with the model, rather than the data.

And because these options are all values we (as machine learning engineers and data scientists) can change, they are referred as hyperparameters.

# 6. The missing piece: non-linearity

"What patterns could you draw if you were given an infinite amount of a straight and non-straight lines?"

Or in machine learning terms, an infinite (but really it is finite) of linear and non-linear functions?

- Linear = straight lines
- Non-linear = non-straight lines

Artificial neural networks are a large combination of linear (straight) and non-straight (non-linear) functions which are potentially able to find patterns in data.

# 8. Putting it all together with a multi-class classification problem

- Binary classification = one thing or another (cat vs. dog, spam vs. not spam, fraud or not fraud)
- Multi-class classification = more than one thing or another (cat vs. dog vs. chicken)