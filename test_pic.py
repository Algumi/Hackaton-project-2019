from matplotlib import pyplot as plt
import mxnet as mx
from gluoncv import model_zoo, data, utils
from PIL import Image


class HumanPic:
    __pic_path = "inp_pic.jpg"
    __net = []
    __GPU = False
    __pic_height = 0
    __pic_width = 0
    __accuracy = 0.7

    def get_thumbnails(self, pic):
        pic.save(self.__pic_path)
        x, orig_img = data.transforms.presets.rcnn.load_test(self.__pic_path)
        self.__pic_width = x.shape[3]
        self.__pic_height = x.shape[2]
        if self.__GPU:
            self.__net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True, ctx=mx.gpu(0))
            return [xx[0].asnumpy() for xx in self.__net(x).as_in_context(mx.gpu(0))]
        else:
            self.__net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
            return [xx[0].asnumpy() for xx in self.__net(x)]

    def get_height(self, pic):
        ids, scores, bboxes, masks = self.get_thumbnails(pic)
        objects = [(x[0], y) for x, y in zip(ids, scores) if y[0] > self.__accuracy]
        tags = list(set(self.__net.classes[int(x[0])] for x in objects))
        if tags.count("person") > 0:
            people_bboxes = [bboxes[int(x[0])] for x in objects if x[0] == 0]
            people_heights = [(x[3] - x[1]) / self.__pic_height for x in people_bboxes]
            av_height = sum(people_heights) / len(people_heights)
            return [av_height, tags]
        else:
            return [0, tags]


def test():
    test_pic = Image.open("test10.jpg")
    test = HumanPic()
    print(test.get_height(test_pic))

test()