# Experiment into fundamental operations in neural networks

This code explores how basic networks (e.g. dense layers) can perform fundamental computer science operations (logical, arithmetic etc)

## Running

```shell
pipenv install
pipenv shell
python -m fops.experiments
```

Example (not correct yet!) output:
```
['task', 'dataset', 'network_type', 'network_depth', 'network_activation', 'accuracy', 'loss']
['equality', 'one_hot', 'dense', 0, 'tanh', 0.0, 0.057277124]
['logical_and', 'one_hot', 'dense', 0, 'tanh', 0.0, 0.010405544]
['logical_or', 'one_hot', 'dense', 0, 'tanh', 0.0, 0.022145728]
['logical_xor', 'one_hot', 'dense', 0, 'tanh', 0.0, 0.020593174]
['elementwise_add', 'one_hot', 'dense', 0, 'tanh', 0.0, 0.022100633]
['elementwise_mul', 'one_hot', 'dense', 0, 'tanh', 0.0, 0.0077557103]
['reduce_sum', 'one_hot', 'dense', 0, 'tanh', 0.0, 0.14761811]
['reduce_max', 'one_hot', 'dense', 0, 'tanh', 0.0, 0.0155320065]
['equality', 'many_hot', 'dense', 0, 'tanh', 0.0, 0.013444532]
['logical_and', 'many_hot', 'dense', 0, 'tanh', 1.0, 0.0]
['logical_or', 'many_hot', 'dense', 0, 'tanh', 1.0, 0.0]
['logical_xor', 'many_hot', 'dense', 0, 'tanh', 1.0, 0.0]
['elementwise_add', 'many_hot', 'dense', 0, 'tanh', 1.0, 0.0]
['elementwise_mul', 'many_hot', 'dense', 0, 'tanh', 1.0, 0.0]
['reduce_sum', 'many_hot', 'dense', 0, 'tanh', 1.0, 0.0]
['reduce_max', 'many_hot', 'dense', 0, 'tanh', 1.0, 0.0]
['equality', 'random', 'dense', 0, 'tanh', 0.0, 0.5564104]
['logical_and', 'random', 'dense', 0, 'tanh', 0.0, 3.3445785]
['logical_or', 'random', 'dense', 0, 'tanh', 0.0, 0.50493354]
['logical_xor', 'random', 'dense', 0, 'tanh', 0.0, 1.4137758]
['elementwise_add', 'random', 'dense', 0, 'tanh', 0.0, 0.6893952]
['elementwise_mul', 'random', 'dense', 0, 'tanh', 0.0, 0.44711256]
['reduce_sum', 'random', 'dense', 0, 'tanh', 0.0, 16232.847]
```