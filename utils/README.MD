# utils
## check_point.py
* instantiate a CheckPoint
Note: will automatically add History obj **defined before** into checkpoint
```python
ckpter = CheckPoint(model=model, optimizer=optimizer,
    path='./ckpt', prefix='Run01', interval=2, save_num=1)
```
* Explicitly bind history to save train/val history
```python
# best practice is to new CheckPoint before new History
# Which will automatically add History obj into checkpoint

ckpter.bind_histories([train_hist, val_hist])
```
* call .chech_on() on each epoch
```python
for epoch in range(10):
    ...
    ckpter.check_on(epoch=epoch, monitor='loss', loss_acc=val_loss_acc)
```
## history.py
a class to hold and plot acc&loss history
* instantiate a History
```python
his = History(name='train')
```
* call `.add()` method
```python
logs = {'loss': 1e-1, 'acc': 0.2}
his.add(epoch=1, log)
```
* plot train/val history
```python
his.plot()
```
## vis.py
* visualize trained **model filters**
* visualize **feature maps** given input and model architecture
* visualize train/val histories
### Plot filters
* only support 1st layer
```python
vis.conv1_weight()
```

### Plot feature map
* chaining input, get feature map and show
```python
# plot batch=0,channel=1
vis.feed(x).fea_map('conv1').show(batch_idx=0, channel_idx=1)
# plot whole batch by setting `batch_idx=None`
vis.feed(x).fea_map('conv1').show(batch_idx=None, channel_idx=1)
```

### Plot train/val histories
```python
vis.phistory()
```