TensorFlow使用数据流图来表示独立操作之间依赖关系的计算。这导致了一个low-level编程模型，您首先在其中定义数据流图，然后创建一个TensorFlow会话以在一组本地和远程设备上运行图的一部分。

## Why dataflow graphs?

数据流是并行计算的常用编程模型。在数据流图中，节点代表计算单位，边代表计算所消耗或产生的数据。例如，在TensorFlow图中，tf.matmul操作将对应于具有两个传入边（要相乘的矩阵）和一个传出边（乘法结果）的单个节点。

TensorFlow利用数据流具有在执行程序时的几个优点：

- 并行。通过使用显式边来表示操作之间的依赖关系，系统很容易识别可以并行执行的操作。
- 分布式执行。通过使用显式边来表示在操作之间流动的值，TensorFlow可以将程序分区到连接到不同机器的多个设备（CPU，GPU和TPU）上。 TensorFlow在设备之间插入必要的通信和协调。
- Compilation。 TensorFlow的XLA编译器可以使用数据流图中的信息生成更快的代码，例如，通过融合相邻操作。
- 可移植性。数据流图是模型中代码的语言无关表示。您可以使用Python构建数据流图，将其存储在SavedModel中，然后在C ++程序中将其恢复为低延迟inference。

## What is a tf.Graph?

一个tf.Graph包含两种相关的信息：

- 图结构。图的节点和边，表示独立操作如何组合在一起，但没有规定应该如何使用它们。图结构就像汇编代码：检查它可以传达一些有用的信息，但它不包含源代码传达的所有有用上下文。
- 图表集合。 TensorFlow提供了一个通用机制来存储tf.Graph中的元数据集合。 tf.add_to_collection函数使您可以将对象列表与键关联（其中tf.GraphKeys定义了一些标准键），而tf.get_collection则可以查找与键关联的所有对象。 TensorFlow库的很多部分都使用这个工具：例如，当你创建一个tf.Variable时，它默认添加到代表“全局变量”和“可训练变量”的集合中。当您稍后创建tf.train.Saver或tf.train.Optimizer时，这些集合中的变量将用作默认参数。

## Building a tf.Graph

大多数TensorFlow程序从数据流图构建阶段开始。在这个阶段，您调用构建新的tf.Operation（node）和tf.Tensor（edge）对象并将它们添加到tf.Graph实例的TensorFlow API函数。 **TensorFlow提供了一个默认图形，它是对同一上下文中所有API函数的隐式参数**。例如：

- 调用tf.constant（42.0）会创建一个tf.Operation，它会生成值42.0，将其添加到默认图形中，并返回一个表示常量值的tf.Tensor。
- 调用tf.matmul（x，y）将创建一个tf.Operation，它将tf.Tensor对象x和y的值相乘，将其添加到默认图形中，并返回一个表示乘法结果的tf.Tensor。
- 执行v = tf.Variable（0）会将一个tf.Operation添加到图中，该操作将存储一个在tf.Session.run调用之间持续存在的可写张量值。 tf.Variable对象封装了这个操作，可以像张量一样使用，它将读取当前存储的值。 tf.Variable对象还具有创建tf.Operation对象的assign和assign_add等方法，该对象在执行时更新存储的值。 （有关变量的更多信息，请参阅变量。）
- 调用tf.train.Optimizer.minimize会将操作和张量添加到计算梯度的默认图形中，并返回一个tf.Operation，在运行时将这些梯度应用于一组变量。

大多数程序仅依赖于默认图形。但是，请参阅[multiple_graphs](https://www.tensorflow.org/programmers_guide/graphs#programming_with_multiple_graphs)以获取更高级用例。高级API（如tf.estimator.Estimator API）会自行管理默认图形，并且 - 例如 - 用于训练和评估过程可能会创建的不同图。

> 注意：调用TensorFlow API中的大多数函数只会将操作和张量添加到默认图形中，但不会执行实际计算。相反，您需要编写这些函数，直到获得表示总体计算的tf.Tensor或tf.Operation（例如执行梯度下降的一个步骤），然后将该对象传递给tf.Session以执行计算。

## Naming operations

tf.Graph对象为其包含的tf.Operation对象定义一个名称空间。 TensorFlow自动为图中的每个操作选择一个唯一名称，但给操作描述性名称可以使您的程序更易于阅读和调试。 TensorFlow API提供了两种覆盖操作名称的方法：

- 每个创建新的tf.Operation或返回新的tf.Tensor的API函数接受一个可选的name参数。例如，tf.constant（42.0，name =“answer”）创建一个名为“answer”的新tf.Operation并返回一个名为“answer：0”的tf.Tensor。如果默认图形已经包含名为“answer”的操作，那么TensorFlow会将“_1”，“_2”等附加到名称上，以使其唯一。
- tf.name_scope函数可以为特定上下文中创建的所有操作添加名称范围前缀。当前名称作用域前缀是“/” - 所有活动的tf.name_scope上下文管理器名称的分隔列表。如果在当前上下文中已经使用了名称范围，则TensorFlow会附加“_1”，“_2”等。例如：

```python
c_0 = tf.constant(0, name="c")  # => operation named "c"

# Already-used names will be "uniquified".
c_1 = tf.constant(2, name="c")  # => operation named "c_1"

# Name scopes add a prefix to all operations created in the same context.
with tf.name_scope("outer"):
  c_2 = tf.constant(2, name="c")  # => operation named "outer/c"

  # Name scopes nest like paths in a hierarchical file system.
  with tf.name_scope("inner"):
    c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"

  # Exiting a name scope context will return to the previous prefix.
  c_4 = tf.constant(4, name="c")  # => operation named "outer/c_1"

  # Already-used name scopes will be "uniquified".
  with tf.name_scope("inner"):
    c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"
```

图形可视化器使用名称范围对操作进行分组，并减少图形的视觉复杂性。

请注意，tf.Tensor对象是以生成张量作为输出的tf.Operation隐式命名的。张量名称的格式为`<OP_NAME>：<i>`其中

- `<OP_NAME>`是生成它的操作的名称。
- `<i>`是一个整数，表示操作输出中该张量的索引。


## 将操作放置在不同的设备上

如果您希望您的TensorFlow程序使用多个不同的设备，则tf.device函数提供了一种便捷的方式来请求将在特定上下文中创建的所有操作放置在同一设备（或设备类型）上。

具有以下形式：

```
/job:<JOB_NAME>/task:<TASK_INDEX>/device:<DEVICE_TYPE>:<DEVICE_INDEX>
```

其中,

- `<JOB_NAME>` 是不以数字开头的字母数字字符串。
- `DEVICE_TYPE` 是已注册的设备类型（例如GPU或CPU）。
- `TASK_INDEX` 是一个非负整数，表示名为<JOB_NAME>的作业中任务的索引。有关作业和任务的说明，请参阅[tf.train.ClusterSpec](https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec)。
- `<DEVICE_INDEX>` 是一个表示设备索引的非负整数，例如，用于区分同一进程中使用的不同GPU设备.

您不需要指定设备规范的每个部分。例如，如果您正在单GPU配置的单机配置中运行，则可以使用tf.device将一些操作固定到CPU和GPU：

```python
# Operations created outside either context will run on the "best possible"
# device. For example, if you have a GPU and a CPU available, and the operation
# has a GPU implementation, TensorFlow will choose the GPU.
weights = tf.random_normal(...)

with tf.device("/device:CPU:0"):
  # Operations created in this context will be pinned to the CPU.
  img = tf.decode_jpeg(tf.read_file("img.jpg"))

with tf.device("/device:GPU:0"):
  # Operations created in this context will be pinned to the GPU.
  result = tf.matmul(weights, img)
```

如果要在`@ {$deploy/distributed$typical distributed configuration}`中部署TensorFlow，则可以指定作业名称和任务ID以将变量放置在参数服务器作业（`/job:ps`）中的任务上，而其他操作放在（`/job:worker`）：

```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(tf.truncated_normal([784, 100]))
  biases_1 = tf.Variable(tf.zeroes([100]))

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(tf.truncated_normal([100, 10]))
  biases_2 = tf.Variable(tf.zeroes([10]))

with tf.device("/job:worker"):
  layer_1 = tf.matmul(train_batch, weights_1) + biases_1
  layer_2 = tf.matmul(train_batch, weights_2) + biases_2
```

tf.device为您提供了很大的灵活性，可以为独立操作或TensorFlow图的广泛施展空间。在很多情况下，有简单的启发式方法很有效。例如，tf.train.replica_device_setter API可以与tf.device配合使用，以执行数据并行分布式训练的操作。例如，以下代码片段显示了tf.train.replica_device_setter如何将不同的放置策略应用于tf.Variable对象和其他操作：

```python
with tf.device(tf.train.replica_device_setter(ps_tasks=3)):
  # tf.Variable objects are, by default, placed on tasks in "/job:ps" in a
  # round-robin fashion.
  w_0 = tf.Variable(...)  # placed on "/job:ps/task:0"
  b_0 = tf.Variable(...)  # placed on "/job:ps/task:1"
  w_1 = tf.Variable(...)  # placed on "/job:ps/task:2"
  b_1 = tf.Variable(...)  # placed on "/job:ps/task:0"

  input_data = tf.placeholder(tf.float32)     # placed on "/job:worker"
  layer_0 = tf.matmul(input_data, w_0) + b_0  # placed on "/job:worker"
  layer_1 = tf.matmul(layer_0, w_1) + b_1     # placed on "/job:worker"
```

## Tensor-like objects

许多TensorFlow操作将一个或多个tf.Tensor对象作为参数。例如，tf.matmul需要两个tf.Tensor对象，而tf.add_n需要一个n维tf.Tensor对象的列表。为了方便起见，这些函数将接受一个类似于张量的对象来代替tf.Tensor，并使用tf.convert_to_tensor方法将其隐式转换为tf.Tensor。类似张量的对象包括以下类型的元素：

- tf.Tensor
- tf.Variable
- numpy.ndarray
- list(和Tensor-like objects列表)
- python 标量: bool, float, int, str

您可以使用tf.register_tensor_conversion_function注册附加张量类型的类型。

> 注意：默认情况下，每次使用相同的Tensor-like对象时，TensorFlow都会创建一个新的tf.Tensor。如果Tensor-like对象很大（例如包含一组训练示例的numpy.ndarray）并且多次使用它，则可能会用完内存。为了避免这种情况，在Tensor-like对象上手动调用tf.convert_to_tensor一次，然后使用返回的tf.Tensor

## 在tf.Session中执行图

---

TensorFlow使用tf.Session类来表示客户端程序（通常是Python程序）之间的连接，尽管类似的接口在其他语言中可用---和C ++运行时。 tf.Session对象提供对本地机器中的设备以及使用分布式TensorFlow运行时的远程设备的访问。它还会缓存关于您的tf.Graph的信息，以便您可以高效地多次运行相同的计算。

### Creating a tf.Session

如果您使用低级别TensorFlow API，则可以为当前默认图创建一个tf.Session，如下所示：

```python
# Create a default in-process session.
with tf.Session() as sess:
  # ...

# Create a remote session.
with tf.Session("grpc://example.org:2222"):
  # ...
```

由于tf.Session拥有物理资源（如GPU和网络连接），因此它通常用上下文管理器（with block）来在退出块时自动关闭会话。也可以在不使用with块的情况下创建一个会话，但是当你完成它以释放资源时，你应该明确地调用tf.Session.close。

> 注意：tf.train.MonitoredTrainingSession或tf.estimator.Estimator等更高级别的API将为您创建和管理tf.Session。这些API接受可选的`target`和`config`参数（直接或者作为tf.estimator.RunConfig对象的一部分），具有与下述相同的含义。

tf.Session.init接受三个可选参数：

- `target` 如果此参数为空（默认值），则会话将仅使用本地计算机中的设备。但是，您也可以指定`grpc//URL`来指定TensorFlow服务器的地址，从而使会话可以访问此服务器控制的计算机上的所有设备。有关如何创建TensorFlow服务器的详细信息，请参阅[tf.train.Server](https://www.tensorflow.org/api_docs/python/tf/train/Server)。例如，在between-graph replication中，tf.Session在与客户端相同的进程中连接到tf.train.Server。[分布式TensorFlow部署](https://www.tensorflow.org/deploy/distributed)指南介绍了其他常见情况。

- `graph` 默认情况下，一个新的tf.Session将被绑定到---并且只能在当前的默认图中运行操作。如果您在程序中使用多个图形（请参阅[使用多个图形编程](https://www.tensorflow.org/programmers_guide/graphs#programming_with_multiple_graphs)以获取更多详细信息），则可以在构建会话时指定显式tf.Graph。

- `config` 该参数允许您指定控制会话行为的tf.ConfigProto。例如，一些配置选项包括：
    - `allow_soft_placement` 将其设置为True以启用“软”设备放置算法，该算法会忽略tf.device尝试将CPU-only操作放置在GPU设备上的操作，并将其放置在CPU上。
    - `cluster_def` 使用分布式TensorFlow时，此选项允许您指定在计算中使用哪些机器，并提供作业名称，任务索引和网络地址之间的映射。有关详细信息，请参阅[tf.train.ClusterSpec.as_cluster_def](https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec#as_cluster_def)。
    - `graph_options.optimizer_options` 提供对TensorFlow在执行图形之前对其进行优化的控制。
    - `gpu_options.allow_growth` 将其设置为True以更改GPU内存分配器，以便逐渐增加分配的内存量，而不是在启动时分配大部分内存。

### 使用tf.Session.run执行操作

tf.Session.run方法是运行tf.Operation或评估tf.Tensor的主要机制。您可以将一个或多个tf.Operation或tf.Tensor对象传递给tf.Session.run，并且TensorFlow将执行计算结果所需的操作。

tf.Session.run要求你指定一个确定返回值的**提取列表**，可以是一个tf.Operation，一个tf.Tensor，或者一个类似Tensor-like的类型，比如tf.Variable。这些提取决定了整个tf.Graph的哪个子图必须执行来产生结果：这是包含在提取列表中指定的所有操作的子图，以及所有其输出用于计算提取值的操作的子图。例如，下面的代码片段显示了tf.Session.run的不同参数如何导致不同的子图被执行：

```python
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
  # Run the initializer on `w`.
  sess.run(init_op)

  # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
  # the result of the computation.
  print(sess.run(output))

  # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
  # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
  # op. Both `y_val` and `output_val` will be NumPy arrays.
  y_val, output_val = sess.run([y, output])
```

tf.Session.run还可以选择性地接收一个feeds的字典，它是从tf.Tensor对象（通常为tf.placeholder tensors）到值（通常是Python标量，列表或NumPy数组）的映射，它们将替换Tensor的值然后执行。例如：

```python
# Define a placeholder that expects a vector of three floating-point values,
# and a computation that depends on it.
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
  # Feeding a value changes the result that is returned when you evaluate `y`.
  print(sess.run(y, {x: [1.0, 2.0, 3.0]}))  # => "[1.0, 4.0, 9.0]"
  print(sess.run(y, {x: [0.0, 0.0, 5.0]}))  # => "[0.0, 0.0, 25.0]"

  # Raises `tf.errors.InvalidArgumentError`, because you must feed a value for
  # a `tf.placeholder()` when evaluating a tensor that depends on it.
  sess.run(y)

  # Raises `ValueError`, because the shape of `37.0` does not match the shape
  # of placeholder `x`.
  sess.run(y, {x: 37.0})
```

tf.Session.run还接受一个可选`options`参数，使您可以指定有关该调用的选项，以及一个可选的`run_metadata`参数，使您可以收集有关执行的元数据。例如，您可以一起使用这些选项来收集有关执行的跟踪信息：

```python
y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
  # Define options for the `sess.run()` call.
  options = tf.RunOptions()
  options.output_partition_graphs = True
  options.trace_level = tf.RunOptions.FULL_TRACE

  # Define a container for the returned metadata.
  metadata = tf.RunMetadata()

  sess.run(y, options=options, run_metadata=metadata)

  # Print the subgraphs that executed on each device.
  print(metadata.partition_graphs)

  # Print the timings of each operation that executed.
  print(metadata.step_stats)
```

## Visualizing your graph

---

TensorFlow包含的工具可以帮助您理解图表中的代码。图形可视化器是TensorBoard的一个组件，它可以在浏览器中直观地呈现图形的结构。创建可视化的最简单方法是在创建tf.summary.FileWriter时传递tf.Graph参数给它：

```python
# Build your graph.
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
# ...
loss = ...
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
  # `sess.graph` provides access to the graph used in a `tf.Session`.
  writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

  # Perform your computation...
  for i in range(1000):
    sess.run(train_op)
    # ...

  writer.close()
```

> 注意：如果您使用的是tf.estimator.Estimator，则图形（和任何摘要）将自动记录到您在创建估计器时指定的model_dir。

然后，您可以打开TensorBoard中的日志，导航到“图形”选项卡，并查看图形结构的高级可视化。请注意，典型的TensorFlow图 - 特别是具有自动计算梯度的训练图 - 具有太多的节点，无法一次显示。图表可视化器利用名称范围将相关操作分组为“超级”节点。您可以点击任何这些超级节点上的橙色“+”按钮来展开里面的子图。

## Programming with multiple graphs

---

> 注意：在训练模型时，组织代码的常用方法是使用一个图形来训练模型，并使用单独的图形来评估或执行训练模型的推理。在许多情况下，推理图将与训练图不同：例如，dropout和batch normalization等技术在每种情况下都使用不同的操作。此外，默认情况下，tf.train.Saver等实用程序使用tf.Variable对象（具有基于tf.Operation的名称）的名称来标识保存的检查点中的每个变量。以这种方式进行编程时，可以使用完全独立的Python进程来构建和执行图形，也可以在同一进程中使用多个图形。本节介绍如何在同一过程中使用多个图。

如上所述，TensorFlow提供了一个“默认图”，该图默认传递给同一上下文中的所有API函数。对于许多应用程序来说，一张图就足够了。但是，TensorFlow还提供了操作默认图的方法，这在更高级的使用情况下可能很有用。例如：

- tf.Graph为tf.Operation对象定义了名称空间：单个图形中的每个操作都必须具有唯一的名称。如果所请求的名称已被使用，TensorFlow将通过在名称后附加“_1”，“_2”等来“独立”操作名称。使用多个显式创建的图可以更好地控制每个操作的名称。
- 默认图表存储有关添加到它的每个tf.Operation和tf.Tensor的信息。如果你的程序创建了大量未连接的子图，使用不同的tf.Graph来构建每个子图可能会更有效率，这样无用不相关的状态可以不收集。

您可以使用tf.Graph.as_default上下文管理器来安装不同的tf.Graph作为默认图形：

```python
g_1 = tf.Graph()
with g_1.as_default():
  # Operations created in this scope will be added to `g_1`.
  c = tf.constant("Node in g_1")

  # Sessions created in this scope will run operations from `g_1`.
  sess_1 = tf.Session()

g_2 = tf.Graph()
with g_2.as_default():
  # Operations created in this scope will be added to `g_2`.
  d = tf.constant("Node in g_2")

# Alternatively, you can pass a graph when constructing a `tf.Session`:
# `sess_2` will run operations from `g_2`.
sess_2 = tf.Session(graph=g_2)

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2
```

要检查当前的默认图形，请调用tf.get_default_graph，它将返回一个tf.Graph对象：

```python
# Print all of the operations in the default graph.
g = tf.get_default_graph()
print(g.get_operations())
```