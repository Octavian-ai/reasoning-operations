# Experiment into fundamental operations in neural networks

This code explores how basic networks (e.g. dense layers) can perform fundamental computer science operations (logical, arithmetic etc)

## Running

```shell
pipenv install
pipenv shell
python -m fops.run_all_experiments
```

Example output in `output.csv`:
```
task,dataset,network_type,network_depth,network_activation,accuracy_pct,accuracy,loss,lr,datetime
concat,one_hot,concat,1,linear,99.0%,0.9921875,0.02309635,1.7782794100389232e-05,2018-11-01 16:46:45
concat,one_hot,dense,1,linear,100.0%,0.9999011,1.4289054e-06,1.7782794100389232e-05,2018-11-01 16:49:17
concat,one_hot,dense,1,tanh,100.0%,1.0,3.806243e-07,1.0000000000000003e-05,2018-11-01 16:52:41
concat,one_hot,dense,1,relu,100.0%,1.0,3.5620099e-07,1.7782794100389232e-05,2018-11-01 16:53:10
concat,one_hot,dense,1,selu,100.0%,0.99797606,2.7229585e-06,1.7782794100389232e-05,2018-11-01 16:55:50
concat,one_hot,dense,1,abs,100.0%,1.0,7.6925426e-07,2.1544346900318827e-05,2018-11-01 17:03:44
concat,one_hot,dense,1,tanh_abs,100.0%,0.9997754,2.0966986e-06,1.7782794100389232e-05,2018-11-01 17:06:23
concat,one_hot,dense,1,sigmoid,100.0%,0.9982654,2.6708472e-06,1.7782794100389232e-05,2018-11-01 17:09:01
concat,one_hot,dense,1,softmax,100.0%,1.0,2.6764098e-07,1.7782794100389232e-05,2018-11-01 17:11:39
concat,one_hot,dense_residual,1,linear,99.0%,0.9921875,0.0032360966,1.7782794100389232e-05,2018-11-01 17:11:56

```