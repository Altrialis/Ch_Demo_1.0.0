# -*- coding: utf-8 -*-
"""

Created on Mon Mar 12 14:01:15 2018
@author: Administrator
目的:
    训练并使其能识别手写中文汉字 3755个(HWDB1.0 手写脱机汉字数据库)
    
"""

import os
import random
import tensorflow.contrib.slim as slim
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
from tensorflow.python.ops import control_flow_ops

"""
    初始化FLAGS数据集
"""
tf.app.flags.DEFINE_integer('charset_size',3755,"设置字符集大小为10")
tf.app.flags.DEFINE_integer('image_size',64,"设置图像大小,x*x的正方形")
tf.app.flags.DEFINE_string('mode','validation','启动模式{"train", "validation", "inference"}')
tf.app.flags.DEFINE_string('dir_train','./HWDB1/train/','训练数据集路径地址')
tf.app.flags.DEFINE_string('dir_test','./HWDB1/test/','测试数据集路径地址')
tf.app.flags.DEFINE_boolean('batch_size', 128, '初始化 batch size,队列批量大小')

tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "是否需要将数据 随机上下翻转")
tf.app.flags.DEFINE_boolean('random_brightness', False, "是否需要将数据 随机调整明暗")
tf.app.flags.DEFINE_boolean('random_contrast', False, "是否需要将数据 随机调整对比度")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', '检查点路径')

tf.app.flags.DEFINE_integer('max_steps', 16002, '最大训练步骤')
tf.app.flags.DEFINE_integer('eval_steps', 100, "计算准确率步骤")
tf.app.flags.DEFINE_integer('save_steps', 500, "存储步骤")

tf.app.flags.DEFINE_string('log_dir', './log', '日志路径')

tf.app.flags.DEFINE_boolean('restore', True, '是否读取检查点信息')

tf.app.flags.DEFINE_string('model_name', 'ZonyHCCR_Demo_1.0.0_model', '模型名称')
FLAGS = tf.app.flags.FLAGS


"""
文本迭代处理类====================================================================
"""
class DataIterator:
    """
    初始化文本迭代处理类
    通过文档路径获得训练数据集
    使用正则表达式补齐5个0
    """
    def __init__(self, data_dir):
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        print(truncate_path)
        self.image_names = []
        """
        用for循环遍历大目录下的文件
        用文件夹名称进行大小比较,如果正确则继续
        """
        for root, sub_folder, file_list in os.walk(data_dir):
            print("目录:" + root)
            if root < truncate_path:
                """
                如果文件夹匹配,遍历获得文件列表
                """
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
            else:
                break
        """
        将获得的文件列表打乱排序,随机排序
        """
        random.shuffle(self.image_names)
        """
        遍历文件列表,通过文件名来生成labels列表
        """
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]
   
    """
    设置类属性
    返回labels的长度,代表训练集的数量
    """
    @property
    def size(self):
        return len(self.labels)

    """
    定义静态方法:数据扩充
    图像随机上下翻转
    图像随机明暗调整
    图像随机对比度调整
    """
    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images
    
    """
    定义对外方法:输入通道
    batch_size:队列批量大小
    """
    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        """
        转换输入输出为 tensorflow对象(tensorflow内核为C所以需要转换成C的数据类型才可以)
        """
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        """
        该方法是从众多tensorlist中随机选取一组tensor
        """
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)
        """
        选取下表为1的数据,即目标矩阵
        """
        labels = input_queue[1]
        """
        通过路径读取图片,tensorflow自带方法(路径为随机选取的tensor中的下表为0的数据,即图片路径)
        """
        images_content = tf.read_file(input_queue[0])
        """
        执行图片归一化,将图片每个像素变为强度值
        """
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        """
        如果需要数据扩充(随机上下浮动,随机明暗浮动,随机对比浮动)则调用该静态方法
        """
        if aug:
            images = self.data_augmentation(images)
        """
        生成一个默认为64*64的矩阵,一会用于转换操作
        """
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        """
        缩放图像
        使用缩放逻辑: 双线性插值
        """
        images = tf.image.resize_images(images, new_size)
        """
        创建队列
        tf.train.shuffle_batch是将队列中数据打乱后，再读取出来，因此队列中剩下的数据也是乱序的，队头也是一直在补充
        capacity = 50000 队列长度
        min_after_dequeue = 10000 出队后保留数
        batch_size = 每次取的队列大小
        """
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,min_after_dequeue=10000)
        """
        返回128大小的batch后数据
        """
        return image_batch, label_batch
"""
===============================================================================
"""

"""
卷积神经网络构造==================================================================
入参 top_k
       k：每个样本的预测结果的前k个最大的数里面是否包含targets预测中的标签，一般都是取1，即取预测最大概率的索引与标签对比。
"""
def build_CNN(top_k):
    """
    创建占位符: keep_prob
    该字段为浮点数,作用是 在DropOut层控制 神经元激活概率
    也就是让某个神经元的激活值以一定的概率keep_prob，让其停止工作，
    这次训练过程中不更新权值，也不参加神经网络的计算。
    但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
    如果值为1则表示一直保持工作(一般预测时才取1)
    """
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    
    """
    创建图片占位符
    四维矩阵
            第一维数量 (一般为Batch后的值,由于batch可配置所以给None不预设值)
            第二维长度 图片的长64
            第三维高度 图片的高64
            第四维深度 图片通道数,灰度图片为1, 假设有多通道(彩色图片) 亦可以识别
    """
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    """
    创建标签占位符(目标矩阵)
    未知长度,一般为batch后的长度
    """
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    """
    创建是否训练布尔标识
    """
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')
    """
    通过silm的arg_scope 定义函数默认值
    定义一些函数的默认参数值，在scope内，我们重复用到这些函数时可以不用把所有参数都写一遍。
    normalizer_fn : 正则化函数,此处使用BN函数, 在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布,加强梯度加速收敛
    normalizer_params : slim.batch_norm中的参数，以字典形式表示 
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
            """
            第一层卷积: 直接输入图像
            入参解析:
                images:指需要做卷积的输入图像
                64:指定卷积核的个数（就是filter的个数）
                [3, 3]:指定卷积和的维度(宽,高)
                1:卷积时的每一步步长
                padding:填充方式,为SAME即会在矩阵外围填充方便卷积/池化等
            """
            conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME',            scope='conv3_1')
            """
            池化层1:
                输入为上一层卷积后的结果
                池化核矩阵
                池化核步骤
            """
            max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME',   scope='pool1')
            """
            第二层卷积:卷积核个数为128个
            """
            conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME',          scope='conv3_2')
            """
            池化层2
            """
            max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME',   scope='pool2')
            """
            第三层卷积:卷积核个数为256个
            """
            conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME',          scope='conv3_3')
            """
            池化层3
            """
            max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME',   scope='pool3')
            """
            第四层卷积:卷积核个数为 512个
            """
            conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME',          scope='conv3_4')
            """
            第五层卷积:卷积核个数为 512个
            """
            conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME',             scope='conv3_5')
            """
            池化层4
            """
            max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME',   scope='pool4')
            """
            输入扁平化
            """
            flatten = slim.flatten(max_pool_4)
            """
            dropout1层
            先对扁平化后的数据集进行 drooupout操作,置部分神经元为不激活状态(slim默认dropout为0.5)
            """
            drop1 = slim.dropout(flatten, keep_prob)
            """
            全链接1层
            全链接, 输入为dropout后的数据, 第二位1024为输出通道数
            激活函数为 ReLU 函数:线性整流函数
            """
            fc1 = slim.fully_connected(drop1, 1024,activation_fn=tf.nn.relu, scope='fc1')
            """
            dropout2层
            """
            drop2 = slim.dropout(fc1, keep_prob)
            """
            输出层输出结果
            默认激活函数为 ReLU
            输出层个数根据FLAGS设定自定义,根据需要识别的汉字个数定义
            """
            logits = slim.fully_connected(drop2, FLAGS.charset_size, activation_fn=None, scope='fc2')
            print("全链接层输出完毕,查看输出的数据类型:",logits)
    """
    softmax层
    第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，
                    单样本的话，大小就是num_classes
    第二个参数labels：实际的标签，大小同上
    """
    softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    """
    求平均值,获得损耗
    """
    loss = tf.reduce_mean(softmax)
    """
    获得准确度
    """
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
    """
    谜之操作,更新损耗
    """
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

    """
    获得当前步骤
    """
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    """
    学习率设置
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    """
    创建训练对象
    """
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
    """
    softmax概率
    """
    probabilities = tf.nn.softmax(logits)

    """
    存入summary 为了可能要的作图
    """
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    """
    top_k准确率  这里一般是 top1置信度
    """
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'is_training': is_training,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}
"""
===============================================================================
"""



"""
训练方法,用于训练中文数据集=========================================================
"""
def train():
    print('----启动训练')
    train_feeder = DataIterator(FLAGS.dir_train) #初始化训练数据集
    test_feeder  = DataIterator(FLAGS.dir_test)  #初始化测试数据集
    model_name = FLAGS.model_name                #保存时的model名称
    print('----训练图像总数',train_feeder.size)
    """
    使用batch随机抓取数据
    """
    train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size,aug=False)
    test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size,aug=False)
    """
    创建TensorflowSession链接
    """
    sess = tf.Session()
    """
    构建神经网络, top_K 为1,返回的置信度只取top_1的结果
    """
    graph = build_CNN(top_k=1)
    """
    创建tf存储对象
    """
    saver = tf.train.Saver()
    """
    加载神经网络模型
    """
    sess.run(tf.global_variables_initializer())
    """
    创建一个协调器
    """
    coord = tf.train.Coordinator()
    """
    通过协调器创建数据读取线程
    """
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
    """
    支持断点续传,默认500步骤保存一次
    """
    start_step = 0
    if FLAGS.restore:
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("读取检查点: {0}".format(ckpt))
            start_step += int(ckpt.split('-')[-1])
    print('----开始训练',start_step)
    """
    异常捕获
    """
    try:
        i = 0 #用i来记录 执行while循环的次数
        if FLAGS.restore:
            i = start_step
        """
        如果协调器未终止,则持续执行入下代码
        """
        while not coord.should_stop():
            i += 1
            print('----协调器执行正常,当前步骤为:' , i)
            start_time = time.time() #记录开始时间
            """
            对象转换,将batch抓取过的对象转成 tensor对象
            """
            train_images_batch, train_labels_batch = sess.run([train_images, train_labels])

            """
            feed字典填充
            """            
            feed_dict = {graph['images']: train_images_batch,   graph['labels']: train_labels_batch,
                         graph['keep_prob']: 0.8,               graph['is_training']: True}
            
            """
            启动训练实例
            """
            _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], 
                     graph['loss'], 
                     graph['merged_summary_op'], 
                     graph['global_step']], feed_dict=feed_dict)
            """
            记录作图用数据
            """
            train_writer.add_summary(train_summary, step)
            end_time = time.time()
            print("步骤: {0} 耗时: {1} 损耗: {2}".format(step, end_time - start_time, loss_val))
            """
            如果训练步骤大于最大步骤终止while循环,否则继续随机抓取
            """
            if step > FLAGS.max_steps:
                break
            """
            如果到了需要测评的步骤,则进行一次准确率评价
            """
            if step % FLAGS.eval_steps == 1:
                """
                转换测试数据为tensor
                """
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                """
                feed字段填充并 启动预测
                """
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['keep_prob']: 1.0,
                             graph['is_training']: False}
                accuracy_test, test_summary = sess.run([graph['accuracy'], graph['merged_summary_op']],
                                                       feed_dict=feed_dict)
                if step > 300:
                    test_writer.add_summary(test_summary, step)
                print('===============预测测试数据集准确度=======================')
                print('第 {0} 步 测试数据集准确率: {1}' .format(step, accuracy_test))
                print('===============训练测试数据集准确度=======================')
            """
            如果到了保存步骤,则执行保存
            """
            if step % FLAGS.save_steps == 1:
                print('保存步骤: {0}'.format(step))
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name),global_step=graph['global_step'])
    except tf.errors.OutOfRangeError:
            """
            执行结束也进行保存
            """
            print('==================训练完成================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name), global_step=graph['global_step'])
    finally:
            coord.request_stop()
    coord.join(threads)
"""
===============================================================================
"""     

"""
预测方法========================================================================
"""
def inference(image,sess,graph):
    print('inference')
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image,
                                                         graph['keep_prob']: 1.0,
                                                         graph['is_training']: False})
    print('预测结果: 下标矩阵 {0} 准确率矩阵 {1}'.format(predict_index,predict_val))


"""
初始化预测方法,载入神经网络图层并从检查点读取数据
"""
def inferenceINI(top=3):
    sess = tf.Session()
    graph = build_CNN(top_k=top)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    return sess,graph
"""
===============================================================================
"""

"""
校验方法,通过(测试数据集+训练数据集)校验模型准确率=================================================
"""
def validation():
    print('----开始校验模型准确率')
    test_feeder = DataIterator(data_dir=FLAGS.dir_test) #将测试数据集与训练数据集一起校验

    final_predict_val = []      #最终预测准确值数组
    final_predict_index = []    #最终预测字下表数组
    groundtruth = []
    """
    创建session链接
    """
    with tf.Session() as sess:
        """
        使用batch随机抓取数据,并初始化图像
        """
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
        """
        创建神经网络模型,top_k准确率取3
        """
        graph = build_CNN(top_k=3)
        """
        启动session实例
        """
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        """
        启动线程
        """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        """
        读取checkpoint保存的神经网络模型
        """
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("读取检查点: {0}".format(ckpt))

        print('----开始预测')
        try:
            i = 0
            acc_top_1, acc_top_k = 0.0, 0.0 #初始化top1准确率与topk准确率
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                """
                将数据转换成tensor对象
                """
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['keep_prob']: 1.0,
                             graph['is_training']: False}
                """
                开始预测
                """
                batch_labels, probs, indices, acc_1, acc_k = sess.run([graph['labels'],
                                                                       graph['predicted_val_top_k'],
                                                                       graph['predicted_index_top_k'],
                                                                       graph['accuracy'],
                                                                       graph['accuracy_top_k']], feed_dict=feed_dict)
                """
                获得返回值
                """
                final_predict_val += probs.tolist()
                final_predict_index += indices.tolist()
                groundtruth += batch_labels.tolist()
                acc_top_1 += acc_1
                acc_top_k += acc_k
                end_time = time.time()
                print("批次: {0} 耗时 {1} 秒, 准确率 = {2}(top_1) {3}(top_k)" .format(i, end_time - start_time, acc_1, acc_k))

        except tf.errors.OutOfRangeError:
            print('==================预测结束================')
            print( FLAGS.batch_size,"||", test_feeder.size);
            acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size
            acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
            print('top_1 准确率 {0} top_k 准确率 {1}'.format(acc_top_1, acc_top_k))
        finally:
            coord.request_stop()
        coord.join(threads) #停止线程?大概
    return {'准确率矩阵': final_predict_val, '下标矩阵': final_predict_index, '事实库': groundtruth}
"""
===============================================================================
"""



"""
main函数,更具Flag中设置的启动模式进行方法调用
辨别启动方式的字段  : FLAGS.mode
启动来源: tf.app.run() 函数
"""
def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":
        print(":::::开始训练::::::")
        train() #调用train函数开始训练
    elif FLAGS.mode == 'validation':
        print(":::::开始验证::::::")
        dct = validation()
        result_file = 'result.dict'
        print('写入结果: {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(dct, f)
        print('写入结束')
    elif FLAGS.mode == 'inference':
        print(":::::开始预测::::::")
        sess,graph = inferenceINI()
      


"""
执行tf.app.run()函数
该函数的作用:
处理flags解析,执行main函数
"""
#if __name__ == "__main__":
#    tf.app.run()
