# Experiment to see how different neural networks perform fundamental reasoning operations

This code explores how basic networks (e.g. dense layers) can perform fundamental reasoning operations (logical, arithmetic etc)

## Running

```shell
pipenv install
pipenv shell
python -m src.run_all_experiments
```

Example output in `output.csv`:
```
task,dataset,network_type,network_depth,network_activation,accuracy_pct,accuracy,loss,lr,global_step,datetime
reduce_sum,one_hot,dense,1,linear,100.0%,1.0,2.3824464e-06,0.01,1000,2018-11-02 16:30:41
reduce_max,one_hot,dense,1,linear,100.0%,1.0,1.1401751e-06,0.01,700,2018-11-02 16:32:11
concat,one_hot,dense,1,linear,99.0%,0.99171096,0.0101305945,0.0001,1000,2018-11-02 16:33:07
dot,one_hot,dense,1,linear,89.0%,0.89275,0.023592463,0.0010000006,100,2018-11-02 16:35:09
elementwise_mul,one_hot,dense,1,linear,99.0%,0.99126565,0.0045539155,0.0001,1000,2018-11-02 16:36:05
one_hot_sum,one_hot,dense,1,linear,99.0%,0.9948965,0.006432534,0.001,200,2018-11-02 16:37:17
elementwise_add,one_hot,dense,1,linear,100.0%,0.9999668,8.8971334e-05,0.01,600,2018-11-02 16:38:44
equality,one_hot,dense,1,linear,100.0%,0.9981777,0.00043973877,0.01,1000,2018-11-02 16:40:14
logical_and,one_hot,dense,1,linear,99.0%,0.9916016,0.0045248806,0.0001,1000,2018-11-02 16:41:08
logical_or,one_hot,dense,1,linear,100.0%,0.9995527,0.00022476114,0.01,600,2018-11-02 16:42:38
logical_xor,one_hot,dense,1,linear,100.0%,0.9978086,0.0005612952,0.01,700,2018-11-02 16:44:07
reduce_sum,one_hot,dense,1,tanh,0.0%,0.0,1.0,1.0,30000,2018-11-02 16:47:25
reduce_max,one_hot,dense,1,tanh,100.0%,1.0,0.0011904258,0.01,1000,2018-11-02 16:48:54
concat,one_hot,dense,1,tanh,99.0%,0.99158007,0.010098731,0.0001,1000,2018-11-02 16:49:49
reduce_sum,random_pos,dense,1,linear,100.0%,1.0,1.7786737e-05,0.01,1000,2018-11-02 16:53:10
reduce_max,random_pos,dense,1,linear,100.0%,1.0,0.00022204228,0.01,1000,2018-11-02 16:53:16
concat,random_pos,dense,1,linear,100.0%,1.0,1.6202648e-09,0.01,1000,2018-11-02 16:53:23
dot,random_pos,dense,1,linear,3.0%,0.0305,14.050371,0.009999993,30000,2018-11-02 16:54:53
elementwise_mul,random_pos,dense,1,linear,43.0%,0.43446094,0.11553457,0.0010000006,30000,2018-11-02 16:56:05
one_hot_sum,random_pos,dense,1,linear,100.0%,0.9962246,0.004065975,0.0010000006,1000,2018-11-02 16:56:12
elementwise_add,random_pos,dense,1,linear,100.0%,1.0,4.6486655e-15,0.01,1000,2018-11-02 16:56:50
equality,random_pos,dense,1,linear,100.0%,0.99992967,0.001491283,0.01,100,2018-11-02 16:56:56
logical_and,random_pos,dense,1,linear,100.0%,1.0,1.4269211e-15,0.01,1000,2018-11-02 16:57:03
logical_or,random_pos,dense,1,linear,100.0%,1.0,1.4921953e-15,0.01,1000,2018-11-02 16:57:09
logical_xor,random_pos,dense,1,linear,100.0%,0.9999375,0.0014065175,0.01,100,2018-11-02 16:57:15
reduce_sum,random_pos,dense,1,tanh,8.0%,0.081,73.873695,0.0010000006,30000,2018-11-02 16:58:48

```