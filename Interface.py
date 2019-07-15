#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Interface
   Description :  问题要素接口文件
   Author :       anlin
   date：          2019-06-17
-------------------------------------------------
   Change Activity:
                   2019-06-17:
-------------------------------------------------
"""
import tensorflow as tf
import numpy as np
import jieba
import os
import re
from gensim.models.word2vec import Word2Vec

sess_config = tf.ConfigProto(allow_soft_placement=True)


# 多标签模型类,用于加载模型以及预测输出
class MultiLabelsModel:
    def __init__(self, labels_set, scene, q_type, root_dir, wv_obj: Word2Vec, vocab):
        """
        :param labels_set: 标签集, 列表形式
        :param scene: 场景
        :param q_type: 问题类型
        :param root_dir: 模型根目录,即ModelOutput
        :param wv_obj: word2vec对象
        :param vocab: 词典
        """
        self.__wv_obj = wv_obj
        self.__sess = tf.Session(graph=tf.Graph(), config=sess_config)
        self.__labels_set = labels_set
        self.__p = re.compile("(\d+_\d+_A_)|([\r\n])|(\s+)")
        self.__vocab = vocab
        self.__word2vec_max_words = 200  # 控制文本最多取多少个词
        # 加载模型以及相关变量或者placeholder
        with self.__sess.as_default():
            with self.__sess.graph.as_default():
                self.__model_saver = tf.train.import_meta_graph(
                    os.path.join(root_dir, "Model", q_type, scene, "model.meta"))
                self.__model_saver.restore(self.__sess, tf.train.latest_checkpoint(
                    os.path.join(root_dir, "Model", q_type, scene)))
                self.__graph = tf.get_default_graph()
                self.__x_input = self.__graph.get_tensor_by_name("x_input:0")
                self.__is_train = self.__graph.get_tensor_by_name("is_training:0")
                self.__logits = self.__graph.get_tensor_by_name("output/fully_connected/BiasAdd:0")
                predicts = tf.nn.sigmoid(self.__logits) - 0.5
                self.__predicts = tf.cast(predicts > 0, dtype=tf.int32)

    def __encoding_x(self, voice_text):
        """
        :param voice_text: 原语音文本
        :return: (1, 200, 200, 1)
        """
        vector = []
        voice_text = re.sub(self.__p, "", voice_text)
        w_count = 0  # 控制为200x200
        for t in jieba.lcut(voice_text):
            if t not in ['您', '你', '的', '了', '请', '哎', '啊', '哦', '嗷', '哈',
                         '哇', '唉', '咦', '是', '我', '这', '那']:
                if t not in self.__vocab:
                    vector.append(self.__wv_obj["UKN"].reshape(self.__wv_obj.vector_size, 1))
                else:
                    vector.append(self.__wv_obj[t].reshape(self.__wv_obj.vector_size, 1))
                w_count += 1
                if w_count >= self.__word2vec_max_words:
                    break
        for _ in range(self.__word2vec_max_words - w_count):
            vector.append(np.zeros(shape=[self.__wv_obj.vector_size, 1], dtype=np.float))
        return np.array([vector])

    def __decoding_y(self, logits):
        """
        :param logits: 独热编码 2-D size but now first dim is 1
        :return: 汉语标签, "|"分隔
        """
        l = []
        for i, j in enumerate(logits[0]):
            if j == 1:
                l.append(self.__labels_set[i])
        return '|'.join(l)

    # 预测输出, 开放接口
    def predict(self, voice_text):
        with self.__sess.as_default():
            with self.__sess.graph.as_default():
                predicts = self.__sess.run(self.__predicts, feed_dict={
                    self.__x_input: self.__encoding_x(voice_text),
                    self.__is_train: False
                })
                return self.__decoding_y(predicts)

    def __del__(self):
        self.__sess.close()


# 细化内容模型类,用于加载细化内容分析模型
class ContentDetailModel:
    def __init__(self, root_dir, vocab):
        """
        :param root_dir: 模型根目录, 及ModelOutput
        """
        self.__sess = tf.Session(graph=tf.Graph(), config=sess_config)
        self.__vocab = vocab
        self.__p = re.compile("(\d+_\d+_A_)")

        with self.__sess.as_default():
            with self.__sess.graph.as_default():
                self.__model_saver = tf.train.import_meta_graph(
                    os.path.join(root_dir, "Model", "内容细化", "model-50000.meta")
                )
                self.__model_saver.restore(self.__sess, tf.train.latest_checkpoint(
                    os.path.join(root_dir, "Model", "内容细化")
                ))
                self.__graph = tf.get_default_graph()
                self.__is_training_holder = self.__graph.get_tensor_by_name("is_training:0")
                self.__batch_size_holder = self.__graph.get_tensor_by_name("batch_size:0")
                self.__keep_prob_holder = self.__graph.get_tensor_by_name("keep_prob:0")
                self.__x_input_holder = self.__graph.get_tensor_by_name("x_input:0")
                self.__x_length = self.__graph.get_tensor_by_name("x_length:0")
                # greedy search
                self.__sample_id = self.__graph.get_tensor_by_name("cond/decoder_1/transpose_1:0")

    # 编码
    def __encoder_x(self, voice_text: str):
        code = []
        x = voice_text.replace("\r", "").replace("\n", "").replace(" ", "")
        sentences = re.sub(self.__p, "A_", x).split("A_")
        for t in sentences:
            if t != "":
                for j in jieba.lcut(t):
                    if j not in ['您', '你', '的', '了', '请', '哎', '啊', '哦', '嗷', '哈',
                                 '哇', '唉', '咦', '是', '我', '这', '那']:
                        if j in self.__vocab:
                            code.append(self.__vocab.index(j))
                        else:
                            code.append(self.__vocab.index("UKN"))
        return code[0:600]  # 不超过600, 因为训练时的控制,避免inference出现OOM等bug

    # 解码
    def __decoder_y(self, sample_id):
        """
        :param sample_id: [batch_size, target_length] ==> batch_size is constant: 1
        :return:
        """
        r_str = ""
        for t in sample_id[0]:
            if t >= len(self.__vocab):
                continue
            else:
                r_str += self.__vocab[t]
        return r_str

    # 预测
    def predict(self, voice_text):
        x = self.__encoder_x(voice_text)
        sample_id = self.__sess.run(
            self.__sample_id,
            feed_dict={
                self.__is_training_holder: False,
                self.__batch_size_holder: 1,
                self.__keep_prob_holder: 1.0,
                self.__x_input_holder: np.array([x]),
                self.__x_length: np.array([len(x)])
            }
        )
        return self.__decoder_y(sample_id)


class InterfaceForProElements:
    # 需要加载各种模型,维护场景编码等
    def __init__(self):
        """
        关键变量含义
        sess_dic: tensorflow会话对象字典,各模型的
        root_dir: 当前的目录
        model_dic: 多标签分类模型
        scene_map: 场景标签映射 编码--> 自定义
        support_scenes: 目前支持多标签的场景--> 自定义
        """
        self.__root_dir = os.path.dirname(__file__)
        self.__scene_map = {
            "XZ-RHXZ": "rhxz",  # 融合新装
            "XZ-DKXZ": "dkxz",  # 单宽新装
            "XZ-DCXZ": "dcxz",  # 单C新装
            "XZ-DTVXZ": "dtvxz",  # 单TV新装
            "CFLH-DCLW": "dclw",  # 单C离网
            "CFLH-RHSJLW": "rhsjlw",  # 融合手机离网
            "CFLH-RHKDLW": "rhkdlw",  # 融合宽带离网
            "CFLH-RHLW": "rhlw",  # 融合离网
            "CFLH-DKLW": "dklw",  # 单宽离网
            "CFLH-RHCF": "rhcf",  # 融合拆分
            "CFLH-ZGWHKWL": "zgwkhwl",  # 中高危客户挽留
            "JZ-FSCPJB": "fscpjb",  # 附属产品加包
            "JZ-ZNZW": "znzw",  # 智能组网
            "XFXY-ZJXY": "zjxy",  # 租机续约
            "XFXY-JF": "jz",  # 缴赠
            "HJHKHGX-SJHX": "sjhx",  # 手机换新
            "HJHKHGX-PZG": "pzg",  # 普转光
            "TCQY-KDJS": "kdts",  # 宽带提速
            "TCQY-BXLLTCQY": "bxltcqz",  # 不限量套餐迁转
            "XSHD-LLHF": "llhf",  # 流量花房
            "XSHD-CJXT": "cjxt",  # 成就系统
            "WTWJ-KDGZ": "kdgz",  # 宽带故障
            "WTWJ-KDWSM": "kdwsm",  # 宽带网速慢
            "WTWJ-YDXHC": "ydwlxhc",  # 移动网络信号差
            "WTWJ-JTWIFIXHC": "jtwifixhc",  # 家庭wifi信号差
            "WTWJ-YJQX": "yjqx"  # 越级倾向
        }
        self.__p = re.compile("(\d+_\d+_A_)")
        self.__support_scenes = {"ydwlxhc", "kdgz", "kdwsm"}
        # jieba自定义词 ==> 需要区分两个jieba分词环境
        for t in ["4g网", "四g网", "4g", "四g", "3g网", "3g", "二三四g", "二三四g网",
                  "三g网", "三g", "5g网", "5g", "五g网", "五g", "两g网", "两g", "2g网", "2g"]:
            jieba.add_word(t, 1000, "hebei_dianxin")
        self.__model_dic = dict()
        wv_obj = Word2Vec.load(os.path.join(self.__root_dir, "ModelOutput", "Model", "text2vec", "all_word2vec200_cbow.m"))
        vocab = list(wv_obj.wv.vocab.keys())
        self.__model_dic["kdgz_phenomenon"] = MultiLabelsModel(
            labels_set=["标准故障-亮红灯", "标准故障-猫正常", "标准故障-其他", "691-密码错误", "691-身份核实失败",
                        "691-欠费、状态不同步",
                        "691-无接入信息", "宽带-打不开网页", "678/651-亮红灯", "678/651-猫正常", "宽带-掉线",
                        "宽带-其他错误提示", "其他",
                        "宽带-线路故障", "宽带-大面积故障"]
            , scene="宽带故障", q_type="问题现象",
            root_dir=os.path.join(self.__root_dir, "ModelOutput"), wv_obj=wv_obj, vocab=vocab)

        self.__model_dic["kdgz_instance"] = MultiLabelsModel(
            labels_set=["宽带网络", "宽带线路", "光纤猫", "服务类", "其他"], scene="宽带故障", q_type="涉及对象",
            root_dir=os.path.join(self.__root_dir, "ModelOutput"), wv_obj=wv_obj, vocab=vocab)

        self.__model_dic["ydwlxhc_phenomenon"] = MultiLabelsModel(
            labels_set=["语音-无信号", "语音-信号弱/不稳定", "语音-有信号无法接通", "语音-单通",
                        "语音-回音/杂音/断续", "语音-掉话", "语音-外省漫入", "语音-本省漫出",
                        "语音-省内漫游质量", "语音-其他", "数据-无信号", "数据-信号弱", "数据-频繁掉线", "数据-网速慢",
                        "数据-有信号无法登录",
                        "数据-网页无法正常打开", "数据-外省漫入", "数据-本省漫出", "数据-省内漫游", "数据-其他", "VOLTE-单通",
                        "VOLTE-有信号无法接通", "VOLTE-无信号",
                        "VOLTE-信号弱/不稳定",
                        "VOLTE-回音/杂音/断续", "VOLTE-掉话", "VOLTE-省内漫游质量", "VOLTE-本省漫出",
                        "VOLTE-外省漫入", "VOLTE-其他"]
            , scene="移动网络信号差", q_type="问题现象",
            root_dir=os.path.join(self.__root_dir, "ModelOutput"), wv_obj=wv_obj, vocab=vocab)

        self.__model_dic["ydwlxhc_instance"] = MultiLabelsModel(
            labels_set=["移动网络", "手机上网", "移动语音", "服务类", "其他"], scene="移动网络信号差", q_type="涉及对象",
            root_dir=os.path.join(self.__root_dir, "ModelOutput"), wv_obj=wv_obj, vocab=vocab)

        self.__model_dic["kdwsm_phenomenon"] = MultiLabelsModel(
            labels_set=["宽带-网速慢", "宽带-打开网页慢", "宽带-带宽速率不够", "宽带-玩网游卡", "宽带-看视频卡",
                        "宽带-下载慢", "宽带-网络延迟高", "宽带-图片信息读取慢", "其他"],
            scene="宽带网速慢", q_type="问题现象", root_dir=os.path.join(self.__root_dir, "ModelOutput"), wv_obj=wv_obj, vocab=vocab)

        self.__model_dic["kdwsm_instance"] = MultiLabelsModel(
            labels_set=["宽带网络", "宽带速率", "服务类", "其他"],
            scene="宽带网速慢", q_type="涉及对象", root_dir=os.path.join(self.__root_dir, "ModelOutput"), wv_obj=wv_obj, vocab=vocab)

        # 基于Bi-LSTM S2S Attention的生成式摘要
        self.__model_dic["content_detail"] = ContentDetailModel(root_dir=os.path.join(self.__root_dir, "ModelOutput"),
                                                                vocab=vocab)

    # 接口函数,获取分析结果,传值为json形式的dict
    """
    请求格式 input_dic:{'content': 一条语音文本, 'env': 场景编码, 'id': 工单id}
    返回格式 output_dic:{'data':[{'instance': 涉及对象, 'nps_element': NPS要素, 'refinement': 内容细化,
                        'phenomenon': 问题现象, 'category': 场景编码, 'id': 工单id}]}
    """

    def get_analysis_result(self, input_dic):
        output_dic = {
            'data': [{
                'instance': '', 'nps_element': '', 'refinement': '', 'phenomenon': '',
                'category': input_dic['env'], 'id': input_dic['id']
            }]
        }
        # 先检查场景标签是否存在且是否是目前模型支持的场景
        if input_dic['env'] not in self.__scene_map.keys() or \
                self.__scene_map[input_dic['env']] not in self.__support_scenes:
            return output_dic
        # 正常场景,继续分析
        scene = self.__scene_map[input_dic['env']]

        output_dic['data'][0]['instance'] = self.__get_instance(scene=scene, voice_text=input_dic['content'])
        output_dic['data'][0]['nps_element'] = self.__get_nps_element()
        output_dic['data'][0]['refinement'] = self.__get_refinement(voice_text=input_dic['content'])
        output_dic['data'][0]['phenomenon'] = self.__get_phenomenon(scene=scene, voice_text=input_dic['content'])

        return output_dic

    # 获取问题现象的结果// 考虑后续模型集成
    def __get_phenomenon(self, scene, voice_text):
        return self.__model_dic["%s_phenomenon" % scene].predict(voice_text)

    # 获取内容细化的结果// 考虑后续模型集成
    def __get_refinement(self, voice_text):
        """
        :param voice_text:
        :return:
        """
        return self.__model_dic["content_detail"].predict(voice_text)

    # 获取涉及对象的结果// 考虑后续模型集成
    def __get_instance(self, scene, voice_text):
        return self.__model_dic["%s_instance" % scene].predict(voice_text)

    # 获取NPS要素的结果(6.18版本只返回网络) // 考虑后续模型集成
    @staticmethod
    def __get_nps_element():
        return "网络"

    def __del__(self):
        pass


if __name__ == "__main__":
    print("Hello HeBei DianXin!")
