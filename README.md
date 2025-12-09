# ContLearn 
This is jax repository of universal CL code. The present code work with CNN/FNN and GNN.
--
## Dependencies
jax
equinox
optax
diffrax
numpy 
pandas
matplotlib
tmux

--
## Continual Learning without Architecture change
 The main branch has code that 
performs continual learning without modifying architecture.  

### Code Execution
sh script.sh

--
## Continual Learning with Architecture change
For code regarding the paper, the effect of architecture to mitigate forgetting in continual learning which seeks to 
change the architecture on the fly while training continually.  

### Code Execution

Please Checkout branch AWBT_code, which can be done by 

git checkout AWBT_code

and execute programs by 

sh script.sh

## Modify parameters
To modify parameters use the json files in the json folder.


