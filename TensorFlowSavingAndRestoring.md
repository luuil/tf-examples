本文档介绍了如何保存和恢复变量和模型.

## Saving and restoring variables

---

TensorFlow变量提供了表示由程序操作的共享持久状态的最佳方式.本节介绍如何保存和恢复变量.请注意, 估算器会自动保存和恢复变量（在model_dir中）.

tf.train.Saver类提供了用于保存和恢复模型的方法. tf.train.Saver构造函数将保存和恢复操作添加到图形中的所有变量或指定列表中的变量. Saver对象提供了运行这些操作的方法, 指定要写入或读取的检查点文件的路径.

保存器将恢复已经在模型中定义的所有变量.如果您在加载模型时不知道如何构建其图形（例如, 如果您正在编写通用程序来加载模型）, 那么请阅读本文后面的"保存和恢复模型概述"一节.

TensorFlow将变量保存在二进制检查点文件中, 粗略地说, 它将变量名称映射为张量值.

### Saving variables

用tf.train.Saver（）创建一个Saver来管理模型中的所有变量.例如, 以下代码片段演示了如何调用tf.train.Saver.save方法将变量保存到检查点文件中：

```python
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)
```

### Restoring variables

tf.train.Saver对象不仅将变量保存到检查点文件, 还能恢复变量.请注意, 当您恢复变量时, 您无需事先对其进行初始化.例如, 以下片段演示如何调用tf.train.Saver.restore方法从检查点文件恢复变量：

```python
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

> 没有名称为"/tmp/model.ckpt"的物理文件.它是为检查点创建的文件名的**前缀**.用户**只能使用前缀而不是物理检查点文件**进行交互.

### Choosing which variables to save and restore

如果您没有将任何参数传递给tf.train.Saver（）, 则保存程序将处理图形中的所有变量.每个变量都以创建变量时传递的名称进行保存.

在检查点文件中显式指定变量名称有时很有用.例如, 您可能已经培训练了一个名为"weights"的变量的模型, 该变量的值要恢复到名为"params"的变量中.

仅保存或恢复模型使用的变量子集有时也很有用.例如, 您可能已经训练了一个五层的神经网络, 现在您想要训练一个具有六层的新模型, 以重用五个已训练层的现有权重.您可以使用保存程序恢复前五层的权重.

您可以通过传递给tf.train.Saver（）构造函数来轻松指定要保存或加载的名称和变量：

- 变量列表（将以它们自己的名字存储）.
- 一个Python字典, 其中键是要使用的名称, 值是要管理的变量.

继续前面显示的保存/恢复示例：

```python
tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  v1.initializer.run()
  saver.restore(sess, "/tmp/model.ckpt")

  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

> 注意
    - 如果需要保存和恢复模型变量的不同子集, 可以根据需要创建任意多个Saver对象.同一个变量可以列在多个保存器对象中;它的值只有在Saver.restore（）方法运行时才会更改.
    - 如果您只在会话开始时恢复模型变量的子集, 则必须为其他变量运行初始化操作.
    - 要检查检查点中的变量, 可以使用[inspect_checkpoint](https://www.github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/tools/inspect_checkpoint.py)库, 特别是print_tensors_in_checkpoint_file函数.
    - 默认情况下, Saver使用每个变量的tf.Variable.name属性的值.但是, 当您创建一个Saver对象时, 您可以选择为检查点文件中的变量选择名称.

### Inspect variables in a checkpoint 检查检查点中的变量

我们可以使用inspect_checkpoint库快速检查检查点中的变量.

继续前面显示的保存/恢复示例：

```python
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]

# print only tensor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]
```


## Overview of saving and restoring models

---

当你想保存和加载变量, 图表和图表的元数据 - 基本上, 当你想保存或恢复你的模型时 - 我们推荐使用SavedModel. SavedModel是一种与语言无关, 可恢复的密封式序列化格式. SavedModel使更高级别的系统和工具能够生成, 使用和变换TensorFlow模型. TensorFlow提供了多种与SavedModel交互的机制, 包括tf.saved_model API, Estimator API和CLI.

## APIs to build and load a SavedModel

---

本节重点介绍用于构建和加载SavedModel的API, 特别是在使用较低级别的TensorFlow API时.

### Building a SavedModel

我们提供了SavedModel构建器的Python实现. SavedModelBuilder类提供了保存多个MetaGraphDef的功能. MetaGraph是一个数据流图, 加上它的相关variables, assets, and signatures. MetaGraphDef是MetaGraph的protocol buffer表示.signature是图的一组的输入和输出.

如果需要将assets保存并写入或复制到磁盘, 则可以在添加第一个MetaGraphDef时提供.如果多个MetaGraphDef与同名assets相关联, 则只保留第一个版本.

每个添加到SavedModel的MetaGraphDef都必须使用用户指定的标签进行标注.这些标签提供了一种方法来识别要加载和恢复的特定MetaGraphDef, 以及共享的一组变量和assets.这些标签通常使用功能性标签（例如, serving或training）对MetaGraphDef进行标注, 并且可选地使用有关device的方面（例如GPU）进行标注区分.

例如, 以下代码提示使用SavedModelBuilder构建SavedModel的典型方法：

```python
export_dir = ...
...
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [tag_constants.TRAINING],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets)
...
# Add a second MetaGraphDef for inference.
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph([tag_constants.SERVING])
...
builder.save()
```

### Loading a SavedModel in Python

SavedModel加载器的Python版本为SavedModel提供加载和恢复功能.加载操作需要以下信息：

- 恢复图形定义和变量的session
- 用于标识要加载的MetaGraphDef的标签
- SavedModel的位置（目录）

加载后, 指定的MetaGraphDef中的变量, 资产和签名的子集将恢复到提供的会话中.

```python
export_dir = ...
...
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)
  ...
```

### Loading a SavedModel in C++

SavedModel加载器的C++版本提供了一个API来从路径加载SavedModel, 同时允许SessionOptions和RunOptions.您必须指定与要加载的图形关联的标签. SavedModel的加载版本被称为SavedModelBundle, 它包含MetaGraphDef和加载它的会话.

```cpp
const string export_dir = ...
SavedModelBundle bundle;
...
LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagTrain},
               &bundle);
```

### Loading and Serving a SavedModel in TensorFlow Serving

您可以使用TensorFlowServing Model Server binary轻松加载SavedModel.请参阅有关如何安装服务器的[说明](https://www.tensorflow.org/serving/setup#installing_using_apt-get), 或者如果您愿意, 可以构建它.

一旦你有模型服务器, 运行它：

```bash
tensorflow_model_server --port=port-numbers --model_name=your-model-name --model_base_path=your_model_base_path
```

将port和model_name标志设置为您选择的值. model_base_path标志预期为模型根目录, 每个版本的模型都位于数字命名的子目录中.如果您只有一个版本的模型, 只需将其放置在子目录中, 如：将模型放入`/tmp/model/0001`, 然后`model_base_path`就为`/tmp/model`

将模型的不同版本存储在公用根目录的数字命名子目录中.例如, 假设根目录是`/tmp/model`.如果您只有一个版本的模型, 请将其存储在`/tmp/model/0001`中.如果您有两个版本的模型, 请将第二个版本存储在`/tmp/model/0002`中, 依此类推.将`--model-base_path`标志设置为根目录（在本例中为`/tmp/model`）. TensorFlow Model Server将使用该根目录中编号最高的子目录中的模型进行serving.

## Standard constants

---

SavedModel为各种用例构建和加载TensorFlow图表提供了灵活性.对于最常见的用例, SavedModel的API在Python和C ++中提供了一组易于重复使用和持续共享工具的常量.

### Standard MetaGraphDef tags

您可以使用多组标签来唯一标识保存在SavedModel中的MetaGraphDef.常用标签的一个子集在以下中指定：

- [python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py)
- [c++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/tag_constants.h)

### Standard SignatureDef constants

SignatureDef是一个protocol buffer , 用于定义图形支持的计算签名.常用的输入键, 输出键和方法名称定义如下：

- [python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)
- [c++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/signature_constants.h)


## Using SavedModel with Estimators

---

在训练了一个Estimator模型之后, 您可能需要从该模型创建一个serving来接受请求并返回结果.您可以在您的计算机上本地运行此类服务, 或者在云中对其进行可扩展部署.

要准备一个训练好的估算器用于serving, 您必须以标准SavedModel格式导出它.本节介绍如何做：

- 指定可提serving的输出节点和相应的[API](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto)（分类, 回归或预测）.
- 将您的模型导出为SavedModel格式.
- 从本地服务器提供serving并请求预测.

### Preparing serving inputs

在training时, input_fn（）摄取数据并准备好供模型使用.在serving时, 类似地, serving_input_receiver_fn（）接受推理请求并为模型做好准备.该函数具有以下目的：

- 将占位符添加到服务系统图形, 这些占位符将被推理请求feed.
- 添加一些将输入数据转换为模型预期张量格式的额外op.

该函数返回一个tf.estimator.export.ServingInputReceiver对象, 该对象将占位符和生成的特征张量打包在一起.

一个典型的模式是推理请求以序列化tf.Examples的形式到达, 所以serving_input_receiver_fn（）创建一个单独的字符串占位符来接收它们.然后, serving_input_receiver_fn（）还负责通过向图中添加一个tf.parse_example操作来解析tf.Examples.

在编写这样的serving_input_receiver_fn（）时, 您必须将解析规范传递给tf.parse_example以告知解析器哪些使用变量以及如何将它们映射到Tensor中.解析规范是由[tf.FixedLenFeature](https://www.tensorflow.org/api_docs/python/tf/FixedLenFeature), tf.VarLenFeature和tf.SparseFeature的组成的字典形式.请注意, 此解析规范不应包含任何label或weight列, 因为这些列在服务时不可用 - 与training的input_fn（）中使用的解析规范相反.

总的来说吗, 就是:

```python
feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}

def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
```

tf.estimator.export.build_parsing_serving_input_receiver_fn实用程序函数为常见情况提供了输入接收器.

> 注意：在使用Predict API和本地服务器训练要在serving中使用的模型时, 不需要解析步骤, 因为该模型将接收原始特征数据.

即使您不需要解析或其他输入处理, 也就是说, 如果服务系统直接feed特征张量, **您仍然必须提供一个serving_input_receiver_fn（）**, 它为该特征张量创建占位符并将其传递. tf.estimator.export.build_raw_serving_input_receiver_fn实用程序提供了此功能.

如果这些实用程序不能满足您的需求, 您可以自由编写自己的serving_input_receiver_fn（）.可能这种情况需要这么做, 如果您的训练input_fn（）合并了一些必须在serving阶段也要用的预处理逻辑.为了减少training-serving乱用的风险, 我们建议将这样的处理封装在一个函数中, 然后将这个函数从input_fn（）和serving_input_receiver_fn（）中调用.

请注意, serving_input_receiver_fn（）也决定签名的输入部分.也就是说, 在编写serving_input_receiver_fn（）时, 必须告诉解析器哪些特征符合预期, 以及如何将它们映射到模型的预期输入.相反, 签名的输出部分由模型确定.

### Performing the export 进行导出

要导出已训练的Estimator, 请使用导出根路径和serving_input_receiver_fn调用tf.estimator.Estimator.export_savedmodel.

```python
estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn)
```

该方法通过首先调用serving_input_receiver_fn（）来获取特征张量, 然后调用此估算器的model_fn（）来基于这些特征生成模型图形来构建新图.它启动一个新的会话, 并且默认情况下会将最近的检查点恢复到它. （如果需要, 可以传递不同的检查点）.最后, 它在给定的export_dir_base（即, `export_dir_base/<timestamp>`）下面创建一个带时间戳的导出目录, 并将SavedModel写入其中, 该SavedModel包含从此Session保存的单个MetaGraphDef.

注意：您有责任清理一些老旧的导出文件.否则, 这些东西会在export_dir_base下累积.

### Specifying the outputs of a custom model

在编写自定义的model_fn时, 您必须填充tf.estimator.EstimatorSpec返回值的`export_outputs`元素.这是描述要在服务期间导出和使用的输出签名的`{name：output}`字典.

在进行单一预测的通常情况下, 该字典包含一个元素, `name`不重要.但在一个multi-headed模型中, 每个head都由这个词典中的一个条目表示.在这种情况下, `name`是选择可用于请求serving对象的标志.

每个输出值必须是ExportOutput对象, 例如tf.estimator.export.ClassificationOutput, tf.estimator.export.RegressionOutput或tf.estimator.export.PredictOutput.

这些输出类型直接映射到[TensorFlow Serving API](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto), 因此确定哪些请求类型将得到遵守.

注意：在multi-headed情况下, 将为从model_fn返回的export_outputs dict的每个元素生成一个SignatureDef, 该元素使用相同的键命名.这些SignatureDef仅在其输出方面有所不同, 如相应的ExportOutput条目所提供.输入始终是由serving_input_receiver_fn提供的输入.推断请求可以按名称指定头部.必须使用signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY命名一个头, 以指示在推理请求未指定哪个SignatureDef时将serving哪个SignatureDef.

### Serving the exported model locally

对于本地部署, 您可以使用[TensorFlow Serving](https://github.com/tensorflow/serving)为您的模型提供服务, 这是一个开源项目, 用于加载SavedModel并将其作为gRPC服务公开.

首先, 安装TensorFlow Serving, 然后构建并运行本地模型服务器, 用上面导出的SavedModel的路径替换`export_dir_base`

```bash
bazel build //tensorflow_serving/model_servers:tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_base_path=export_dir_base
```

现在您有一台服务器在端口9000上通过gRPC监听推理请求！

### Requesting predictions from a local server

根据[PredictionService](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto#L15) gRPC API服务定义来响应gRPC请求. （嵌套protocol buffer在各种[相邻文件](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis)中定义）.

从API服务定义中, gRPC框架以各种语言生成客户端库, 以提供对API的远程访问.在使用Bazel构建工具的项目中, 这些库是自动构建的, 并通过这些依赖项提供（例如使用Python）：

```python
  deps = [
    "//tensorflow_serving/apis:classification_proto_py_pb2",
    "//tensorflow_serving/apis:regression_proto_py_pb2",
    "//tensorflow_serving/apis:predict_proto_py_pb2",
    "//tensorflow_serving/apis:prediction_service_proto_py_pb2"
  ]
```

Python客户端代码可以导入这些库：

```python
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
```

注意：prediction_service_pb2将服务定义为一个整体, 因此始终是必需的.但是, 典型的客户端只需要一个classification_pb2, regression_pb2和predict_pb2, 具体取决于所做的请求类型.

发送gRPC请求然后通过组装包含请求数据的protocol buffer并将其传递给服务stub来完成.请注意请求protocol buffer创建时为空, 然后怎样通过[生成的protocol buffer API](https://developers.google.com/protocol-buffers/docs/reference/python-generated)填充.

```python
from grpc.beta import implementations

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = classification_pb2.ClassificationRequest()
example = request.input.example_list.examples.add()
example.features.feature['x'].float_list.value.extend(image[0].astype(float))

result = stub.Classify(request, 10.0)  # 10 secs timeout
```

本例中返回的结果是ClassificationResponse protocol buffer.

这是一个简单的例子; 请参阅[Tensorflow Serving文档](https://www.tensorflow.org/deploy/index)和[示例](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example)以获取更多详细信息.

> 注意：ClassificationRequest和RegressionRequest包含tensorflow.serving.Input protocol buffer, 该protocol buffer又包含tensorflow.Example  protocol buffer.相反, PredictRequest包含从要素名称到通过TensorProto编码的值的映射.相应地：当使用Classify和Regress API时, TensorFlow Serving将序列化的tf.Examples提供给图, 所以你的serving_input_receiver_fn（）应该包含一个tf.parse_example（）Op.但是, 当使用通用预测API时, TensorFlow Serving将原始要素数据提供给图形, 因此应该使用serving_input_receiver_fn（）.

## CLI检查并执行SavedModel

---

您可以使用SavedModel命令行界面（CLI）检查并执行SavedModel.例如, 您可以使用CLI检查模型的SignatureDefs. CLI使您能够快速确认输入张量dtype和形状与模型匹配.此外, 如果您想测试模型, 可以使用CLI通过以各种格式传递示例输入（例如, Python表达式）, 然后获取输出来进行健全性检查.

### Installing the SavedModel CLI

一般来说, 您可以通过以下两种方式之一安装TensorFlow：

- 通过安装预构建的TensorFlow二进制文件.
- 通过从源代码构建TensorFlow.

如果您通过预构建的TensorFlow二进制文件安装了TensorFlow, 则系统中已经在路径名`bin\saved_model_cli`上安装了SavedModel CLI.

如果您从源代码构建TensorFlow, 则必须运行以下附加命令来构建saved_model_cli：

```bash
bazel build tensorflow/python/tools:saved_model_cli
```

### Overview of commands

SavedModel CLI在SavedModel中的MetaGraphDef上支持以下两个命令：

- `show`, 它显示SavedModel中的MetaGraphDef计算.
- `run`, 在MetaGraphDef上运行计算.

### `show` command

SavedModel包含一个或多个MetaGraphDefs, 由它们的标记集标识.为了serve model, 您可能想知道每个模型中的SignatureDefs是什么类型, 以及它们的输入和输出是什么. show命令可让您按层次顺序检查SavedModel的内容.语法如下：

```bash
usage: saved_model_cli show [-h] --dir DIR [--all]
[--tag_set TAG_SET] [--signature_def SIGNATURE_DEF_KEY]
```

例如, 以下命令显示SavedModel中所有可用的MetaGraphDef标记集：

```python
saved_model_cli show --dir /tmp/saved_model_dir
The given SavedModel contains the following tag-sets:
serve
serve, gpu
```

以下命令显示MetaGraphDef中所有可用的SignatureDef键：

```bash
saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve
The given SavedModel `MetaGraphDef` contains `SignatureDefs` with the
following keys:
SignatureDef key: "classify_x2_to_y3"
SignatureDef key: "classify_x_to_y"
SignatureDef key: "regress_x2_to_y3"
SignatureDef key: "regress_x_to_y"
SignatureDef key: "regress_x_to_y2"
SignatureDef key: "serving_default"
```

如果MetaGraphDef在标签集中有多个标签, 则必须指定所有标签, 每个标签用逗号分隔.例如：

```python
saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve,gpu
```

要显示特定SignatureDef的所有输入和输出的TensorInfo, 请将SignatureDef键传递给signature_def选项.当你想知道张量键值, 输入张量的dtype和shape以便稍后执行计算图时, 这非常有用.例如：

```python
saved_model_cli show --dir \
/tmp/saved_model_dir --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
inputs['x'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: x:0
The given SavedModel SignatureDef contains the following output(s):
outputs['y'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: y:0
Method name is: tensorflow/serving/predict
```

要显示SavedModel中的所有可用信息, 请使用--all选项.例如：

```python
saved_model_cli show --dir /tmp/saved_model_dir --all
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['classify_x2_to_y3']:
The given SavedModel SignatureDef contains the following input(s):
inputs['inputs'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: x2:0
The given SavedModel SignatureDef contains the following output(s):
outputs['scores'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: y3:0
Method name is: tensorflow/serving/classify

    ...

signature_def['serving_default']:
The given SavedModel SignatureDef contains the following input(s):
inputs['x'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: x:0
The given SavedModel SignatureDef contains the following output(s):
outputs['y'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: y:0
Method name is: tensorflow/serving/predict
```

### run command

调用运行命令以运行图计算, 传递输入, 然后显示（并可选地保存）输出.语法如下：

```bash
usage: saved_model_cli run [-h] --dir DIR --tag_set TAG_SET --signature_def
                           SIGNATURE_DEF_KEY [--inputs INPUTS]
                           [--input_exprs INPUT_EXPRS] [--outdir OUTDIR]
                           [--overwrite] [--tf_debug]
```

run命令提供以下两种方式将输入传递给模型：

- `--inputs` 选项可以让你在文件中传递numpy ndarray.
- `--input_exprs` 选项使您可以传递Python表达式.
- `--input_examples` 选项使您可以传递tf.train.Example.

### `--inputs`

要在文件中传递输入数据, 请指定`--inputs`选项, 该选项采用以下一般格式：

```bash
--inputs <INPUTS>
```

INPUTS是以下格式之一：

- `<input_key>=<filename>`
- `<input_key>=<filename>[<variable_name>]`

您可能会传递多个INPUTS.如果您确实要传递多个输入, 请使用分号分隔每个输入.

saved_model_cli使用numpy.load加载文件名.文件名可以是以下任何一种格式：

- `.npy`
- `.npz`
- pickle format

.npy文件总是包含一个numpy的ndarray.因此, 从.npy文件加载时, 内容将直接分配给指定的输入张量.如果用该.npy文件指定variable_name, 则variable_name将被忽略并发出警告.

从.npz（zip）文件加载时, 您可以选择指定一个variable_name来标识zip文件中用于加载输入张量key的变量.如果您未指定variable_name, 则SavedModel CLI将检查zip文件中是否只包含一个文件, 并将其加载到指定的输入张量key.

从pickle文件加载时, 如果方括号中没有指定variable_name, 那么pickle文件中的内容将被传递到指定的输入张量键.否则, SavedModel CLI会假定一个字典存储在pickle文件中, 并且将使用与variable_name对应的值.

### `--input_exprs`

要通过Python表达式传递输入, 请指定--input_exprs选项.这对于没有数据文件的情况非常有用, 但仍希望通过一些与模型的SignatureDefs的dtype和shape匹配的简单输入来检查模型.例如：

```bash
<input_key>=[[1],[2],[3]]
```

除了Python表达式之外, 您还可以传递numpy函数.例如：

```bash
<input_key>=np.ones((32,32,3))
```

>（请注意, numpy模块已经可以作为np使用.）

### `--inputs_examples`

要将tf.train.Example作为输入传递, 请指定--input_examples选项.对于每个输入键, 它都需要一个字典列表, 其中每个字典都是tf.train.Example的一个实例.字典键是特征, 值是每个特征的值列表.例如：

```python
`<input_key>=[{"age":[22,24],"education":["BS","MS"]}]`
```

### Save Output

默认情况下, SavedModel CLI将输出写入标准输出.如果一个目录传递给--outdir选项, 则输出将被保存为在指定目录下以输出张量键命名的npy文件.

使用`--overwrite`来覆盖现有的输出文件.

### TensorFlow调试器（tfdbg）集成

如果设置了`--tf_debug`选项, 则SavedModel CLI将在运行SavedModel时使用TensorFlow调试器（tfdbg）观察中间张量和运行时图或子图.


### Full examples of run

给定：

- 您的模型只需添加x1和x2即可获得输出y.
- 模型中的所有张量都具有形状（-1, 1）.
    - 你有两个npy文件： `/tmp/my_data1.npy`, 其中包含numpy ndarray `[[1], [2], [3]]`.
    - `/tmp/my_data2.npy`, 其中包含另一个numpy ndarray `[[0.5], [0.5], [0.5]]`.

要通过模型运行这两个npy文件以获取输出y, 请发出以下命令：

```bash
saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npy;x2=/tmp/my_data2.npy \
--outdir /tmp/out
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
```

让我们稍微改变一下前面的例子.这一次, 而不是两个.npy文件, 你现在有一个.npz文件和一个pickle文件.此外, 你想覆盖任何现有的输出文件.这里是命令：

```bash
saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y \
--inputs x1=/tmp/my_data1.npz[x];x2=/tmp/my_data2.pkl --outdir /tmp/out \
--overwrite
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
```

要使用TensorFlow调试器运行模型, 请发出以下命令：

```bash
saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def serving_default --inputs x=/tmp/data.npz[x] --tf_debug
```

## Structure of a SavedModel directory

---

当您以SavedModel格式保存模型时, TensorFlow会创建一个由以下子目录和文件组成的SavedModel目录：

```txt
assets/
assets.extra/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb|saved_model.pbtxt
```

其中：

- `assets` 是包含辅助（外部）文件（如词汇表）的子文件夹.assets被复制到SavedModel位置, 并且可以在加载特定的MetaGraphDef时读取.
- `assets.extra` 是一个子文件夹, 其中较高级别的库和用户可以添加自己的assets, 该assets与模型共存, 但不会被图加载.这个子文件夹不是由SavedModel库管理的.
- `variables` 是一个包含tf.train.Saver输出的子文件夹. saved_model.pb或saved_model.pbtxt是SavedModel protocol buffer.它包含图形定义为MetaGraphDef protocol buffer.

一个SavedModel可以表示多个图.在这种情况下, SavedModel中的所有图都共享一组检查点（变量）和assets.例如, 下图显示了一个包含三个MetaGraphDefs的SavedModel, 其中三个共享同一组检查点和assets：

![SavedModel](https://www.tensorflow.org/images/SavedModel.svg)

每个图形都与一组特定的标签相关联, 这些标签可以在加载或恢复操作期间进行识别.