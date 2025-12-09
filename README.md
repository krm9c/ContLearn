# ContLearn 

## Continual Learning without Architecture change
This is jax repository of universal CL code. The present code work with CNN/FNN and GNN. The main branch has code that 
performs continual learning without modifying architecture.  

# Dependencies
jax
equinox
diffrax

### Code Execution
sh script.sh


## Continual Learning with Architecture change
For code regarding the paper, the effect of architecture to mitigate forgetting in continual learning which seeks to 
change the architecture on the fly while training continually.  Please Checkout branch AWBT_code, which can be done by 

### Code Execution
git checkout AWBT_code
sh script.sh

## Modify parameters
To modify parameters use the json files in the json folder.


