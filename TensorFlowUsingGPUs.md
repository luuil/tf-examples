## Supported devices

---

在典型的系统中，有多个计算设备。在TensorFlow中，支持的设备类型是CPU和GPU。它们被表示为字符串。例如：

- `/cpu:0`：你机器的CPU。
- `/device:GPU:0`：你机器的GPU，如果你有的话。
- `/device:GPU:1`：您机器的第二个GPU等

如果TensorFlow操作同时具有CPU和GPU，则在将操作分配给设备时，GPU设备将被赋予优先权。例如，`matmul`有CPU和GPU两种实现。在设备`cpu:0`和`gpu:0`的之间，将选择`gpu:0`来运行`matmul`。

## Logging Device placement 输出设备位置

---

要找出您的操作和张量分配给哪些设备，请创建log_device_placement配置选项设置为True的会话。

```python
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```

您应该看到以下输出：

```bash
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/device:GPU:0
a: /job:localhost/replica:0/task:0/device:GPU:0
MatMul: /job:localhost/replica:0/task:0/device:GPU:0
[[ 22.  28.]
 [ 49.  64.]]
```

## 手动指定设备放置

---

如果您希望特定的操作在您选择的设备上运行，而不是自动为您选择的设备运行，则可以使用tf.device创建设备上下文，以便该上下文中的所有操作都具有相同的设备分配。

```python
# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```

你会看到现在a和b被分配给`cpu:0`。由于未明确指定设备用于MatMul操作，因此TensorFlow运行时将根据操作和可用设备（本例中为`gpu:0`）选择一个设备，并根据需要自动复制设备之间的张量。

## Allowing GPU memory growth

---

默认情况下，TensorFlow将几乎所有GPU的GPU内存（受CUDA_VISIBLE_DEVICES影响）映射到进程。通过减少内存碎片，可以更有效地使用设备上相对宝贵的GPU内存资源。

在某些情况下，进程只需要分配可用内存的一个子集，或者仅根据进程需要增加内存使用量。 TensorFlow在Session上提供了两个Config选项来控制这个选项。

第一种是`allow_growth`选项，它试图根据运行时分配只分配尽可能多的GPU内存：它开始分配很少的内存，并且随着Sessions运行并需要更多的GPU内存，我们会增加所需的GPU内存区域以满足tf运行要求。**请注意，我们不释放内存，因为这可能导致更糟的内存碎片**。要打开此选项，请通过以下方式在ConfigProto中设置选项：

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
```

第二种方法是 `per_process_gpu_memory_fraction` 选项，它决定了每个可见GPU应该分配的总内存量的一部分。例如，您可以通过以下方式告诉TensorFlow仅分配每个GPU的总内存的40％：

```python
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
```

如果要真正限制TensorFlow进程可用的GPU内存量，这非常有用。

## Using a single GPU on a multi-GPU system

如果您的系统中有多个GPU，则默认情况下将选择具有最低ID的GPU。如果您想在不同的GPU上运行，则需要明确指定首选项：

```python
# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```

如果您指定的设备不存在，您将得到InvalidArgumentError：

如果您希望TensorFlow在指定的设备不存在的情况下自动选择现有的受支持设备来运行操作，则可以在创建会话时在配置选项中将allow_soft_placement设置为True。

```python
# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(sess.run(c))
```

## Using multiple GPUs

如果您想要在多个GPU上运行TensorFlow，则可以采用多塔式方式(multi-tower fashion)构建模型，其中每个塔都分配有不同的GPU。例如：

```python
# Creates a graph.
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))
```

```python
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K20m, pci bus
id: 0000:02:00.0
/job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla K20m, pci bus
id: 0000:03:00.0
/job:localhost/replica:0/task:0/device:GPU:2 -> device: 2, name: Tesla K20m, pci bus
id: 0000:83:00.0
/job:localhost/replica:0/task:0/device:GPU:3 -> device: 3, name: Tesla K20m, pci bus
id: 0000:84:00.0
Const_3: /job:localhost/replica:0/task:0/device:GPU:3
Const_2: /job:localhost/replica:0/task:0/device:GPU:3
MatMul_1: /job:localhost/replica:0/task:0/device:GPU:3
Const_1: /job:localhost/replica:0/task:0/device:GPU:2
Const: /job:localhost/replica:0/task:0/device:GPU:2
MatMul: /job:localhost/replica:0/task:0/device:GPU:2
AddN: /job:localhost/replica:0/task:0/cpu:0
[[  44.   56.]
 [  98.  128.]]
```