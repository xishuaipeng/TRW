import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2
import shutil
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.patches as patches
from PIL import Image, ImageFont, ImageDraw
from os import listdir, getcwd
class DataSet:
    data_id = '0003' #data set ID like 0001
    analysis_path = '' # the dir contains results file
    @staticmethod
    def extract_classes(class_path):
        class_label = []
        class_file = open(class_path)
        label = class_file.readline()
        while label:
            label = label.strip()
            class_label.append(label)
            label = class_file.readline()
        print(class_label)
        return class_label

    def convert_coordinate(size,  box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    @classmethod
    def __init__(self, data_id):
        self.data_id = data_id
        cur_path = getcwd()
        self.img_path = '%s/%s/JPEGImages'% (cur_path, self.data_id)
        self.annotation_path = '%s/%s/Annotations'% (cur_path, self.data_id)
        self.label_path = '%s/%s/labels'% (cur_path, self.data_id)
        self.analysis_path = '%s/analysis/%s/croped' % (cur_path, self.data_id)
        self.resize_path = '%s/analysis/%s/sized' % (cur_path, self.data_id)
        self.image_index = -1
        self.match_data = []
        self.unmatch_data = []
        self.img_width = 0
        self.img_height = 0
        self.unmatch_index = 0

    def draw_location_map(self):
        # show position
        fig1 = plt.figure()
        plt.xlabel('Image Width')
        plt.ylabel('Image Height')
        plt.axis([0, self.img_width, self.img_height, 0])
        plt.grid(True)
        # plt.title('pedestrian position for detection ')
        plt.plot(self.match_data[:, 1] * self.img_width, self.match_data[:, 2] * self.img_height, 'b^',
                 label=' Location of correct detection ')
        plt.plot(self.unmatch_data[:, 1] * self.img_width, self.unmatch_data[:, 2] * self.img_height, 'r^',
                 label=' Location of miss detection')
        legend = plt.legend(loc='upper center', shadow=True, fontsize='xx-large')
        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('#00FFCC')
        fig1.savefig('./analysis/%s_position.jpg' % self.data_id, dpi=90, bbox_inches='tight')

    def draw_size_map(self):
        fig2 = plt.figure()
        plt.xlabel('Bounding box Width')
        plt.ylabel('Bounding box Height')
        plt.axis([0, self.img_width, 0, self.img_height])
        plt.grid(True)
        # plt.title('pedestrian size for detection ')
        plt.plot(self.match_data[:, 3] * self.img_width, self.match_data[:, 4] * self.img_height, 'b^', label='Size of correct detection')
        plt.plot(self.unmatch_data[:, 3] * self.img_width, self.unmatch_data[:, 4] * self.img_height, 'r^', label='Size of miss detection')
        legend = plt.legend(loc='upper center', shadow=True, fontsize='xx-large')
        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('#00FFCC')
        fig2.savefig('./analysis/%s_size.jpg' % self.data_id, dpi=90, bbox_inches='tight')

    def draw_sizeDistri(self):
        fig = plt.figure()
        area = np.zeros(self.unmatch_index)
        print(area)
        i = 0
        for item in self.unmatch_data:
            area[i] = item[3] * self.img_width * item[4] * self.img_height
            if area[i] > 5000:
                area[i] = 1
            i += 1
        print(area)
        num_bins = 200
        n, bins, patches = plt.hist(area, num_bins, facecolor='blue', alpha=0.5)
        plt.subplots_adjust(left=0.15)
        plt.xlabel("Size of boundingBox")
        plt.ylabel("Quantity")
        plt.show()
        fig.savefig('./analysis/%s_sizeDistribution.jpg' % self.data_id)

    def draw_missDistri(self):
        x = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
        y = [0] * 10
        fig = plt.figure()

        for item in self.unmatch_data:
            center = item[1] * self.img_width
            print(center)
            for i in range(10):
                if center < i * 100:
                    y[i-1] += 1
                    break

        idx = np.arange(len(x))
        plt.bar(idx, y, 1, align="edge", color="blue")
        for x1, y1 in zip(idx, y):
            plt.text(x1+0.2, y1, '%d' % y1, ha='left', va='bottom')
        plt.xticks(idx, x)
        plt.xlabel("Image Width")
        plt.ylabel("Number of missed detection")
        plt.show()
        fig.savefig('./analysis/%s_missChart.jpg' % self.data_id)

    def draw_boundingbox(self):
        fig3 = plt.figure()
        fig3_ax = fig3.add_subplot(111)
        plt.xlabel('Image Width')
        plt.ylabel('Image Height')
        fig3_ax.set_xlim(0, self.img_width)
        fig3_ax.set_ylim(self.img_height, 0)
        plt.grid(True)
        legend_show = True
        for item in self.match_data:
            left = (max((item[1] - item[3] / 2), 0))
            top = (max((1 - item[2] - item[4] / 2), 0))
            if legend_show:
                legend_show = False
                rec = patches.Rectangle((left, top), item[3], item[4], fill=False,
                                        transform=fig3_ax.transAxes, clip_on=False, color='b',
                                        label='Bounding Box of correct detection')
            else:
                rec = patches.Rectangle((left, top), item[3], item[4], fill=False,
                                        transform=fig3_ax.transAxes, clip_on=False, color='b')
            fig3_ax.add_patch(rec)
        legend_show = True
        for item in self.unmatch_data:
            left = (max((item[1] - item[3] / 2), 0))
            top = (max((1 - item[2] - item[4] / 2), 0))
            if legend_show:
                legend_show = False
                rec = patches.Rectangle(
                    (left, top), item[3], item[4], fill=False, transform=fig3_ax.transAxes,
                    clip_on=False, color='r', label='Bounding Box of miss detection')
            else:
                rec = patches.Rectangle(
                    (left, top), item[3], item[4], fill=False, transform=fig3_ax.transAxes,
                    clip_on=False, color='r')
            fig3_ax.add_patch(rec)
        legend = plt.legend(loc='upper center', shadow=True, fontsize='xx-large')
        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('#00FFCC')
        fig3.savefig('./analysis/%s_size_position.jpg' % self.data_id, dpi=90, bbox_inches='tight')

    def draw_example(self):
        if self.image_index == -1 :
            return
        image = '%s/%d.jpg' % (self.resize_path, self.image_index )
        image = Image.open(image)
        fig4, ax4 = plt.subplots(1)
        ax4.imshow(image)
        legend_show = True
        for item in self.match_data:
            left = np.int32(max((item[1] - item[3] / 2), 0) * self.img_width)
            top = np.int32((max(((item[2]) - item[4] / 2), 0)) * self.img_height)
            # right = np.int32(max((match_data[i, 1] + match_data[i, 3] / 2), 0) * img_width)
            # bottom = np.int32(max((match_data[i, 2] + match_data[i, 4] / 2), 0) * img_height)
            if legend_show:
                legend_show = False
                rect = patches.Rectangle((left, top), item[ 3] * self.img_width, item[4] * self.img_height,
                                         linewidth=1, edgecolor='b', facecolor='none',
                                         label='Bounding Box of correct detection')
            else:
                rect = patches.Rectangle((left, top), item[ 3] * self.img_width, item[4] * self.img_height,
                                         linewidth=1, edgecolor='b', facecolor='none')

            ax4.add_patch(rect)
        legend_show = True
        for item in self.unmatch_data:
            left = np.int32(max((item[1] - item[3] / 2), 0) * self.img_width)
            top = np.int32((max(((item[2]) - item[4] / 2), 0)) * self.img_height)
            # right = np.int32(max((unmatch_data[i, 1] + unmatch_data[i, 3] / 2), 0) * img_width)
            # bottom = np.int32(max((unmatch_data[i, 2] + unmatch_data[i, 4] / 2), 0) * img_height)
            if legend_show:
                legend_show = False
                rect = patches.Rectangle((left, top), item[3] * self.img_width, item[4] * self.img_height,
                                         linewidth=1, edgecolor='r', facecolor='none',
                                         label='Bounding Box of miss detection')
            else:
                rect = patches.Rectangle((left, top), item[3] * self.img_width, item[4] * self.img_height, linewidth=1, edgecolor='r',
                                         facecolor='none')

            ax4.add_patch(rect)
        legend = plt.legend(loc='upper center', shadow=True, fontsize='xx-large')
        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('#00FFCC')
        fig4.savefig('./analysis/%s_example.jpg' % self.data_id, dpi=90, bbox_inches='tight')

    def extract_classes(class_path):
        class_label = []
        class_file = open(class_path)
        label = class_file.readline()
        while label:
            label = label.strip()
            class_label.append(label)
            label = class_file.readline()
        print(class_label)
        return class_label

    def generate_label(self):
        anno_dir = self.annotation_path
        txt_dir = self.label_path
        img_dir = self.img_path
        if os.path.isdir(txt_dir):
            shutil.rmtree(txt_dir)
        os.makedirs(txt_dir)
        class_path = 'class.name'
        list_path = './%s/all.list' % self.data_id
        xml_list = listdir(anno_dir)
        classes = DataSet.extract_classes(class_path)
        list_file = open(list_path, 'w')
        for i in xml_list:
            n_object = 0
            xml_path = os.path.join(anno_dir, i)
            img_path = os.path.join(img_dir, '%s.jpg' % i[:-4])
            if os.path.isfile(img_path) is False:
                print('%s is not exit' % img_path)
                continue
            # print(xml_path)
            txt_path = os.path.join(txt_dir, '%s.txt' % i[:-4])
            # print(txt_path)
            xml_file = open(xml_path)
            txt_file = open(txt_path, 'w')
            # print(txt_path)
            # READ XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            # txt_file = open(txt_path, 'w')
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                n_object = n_object + 1
                cls_id = classes.index(cls)
                xml_box = obj.find('bndbox')
                b = (
                float(xml_box.find('xmin').text), float(xml_box.find('xmax').text), float(xml_box.find('ymin').text),
                float(xml_box.find('ymax').text))
                bb = DataSet.convert_coordinate((w, h), b)
                txt_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            txt_file.close()
            if n_object > 0:
                list_file.write(img_path + '\n')
        list_file.close()

    def display_analysis(self, dir_path='', image_index=-1):
        if len(dir_path) == 0:
            dir_path = self.analysis_path
        if image_index != -1:
            self.image_index = image_index
        self.img_width = 928
        self.img_height = 800
        # data preparation
        img_path = os.listdir(dir_path)
        case_num = len(img_path)
        match_data = np.zeros([case_num, 5], np.float32)

        unmatch_data = np.zeros([case_num, 5], np.float32)
        match_index = 0
        unmatch_index = 0

        for img_file in img_path:
            #img_data = cv2.imread(img_file)
            name_resolve = img_file.split('_')
            img_index = np.int32(name_resolve[0])
            if (img_index != image_index & image_index != -1):
                continue
            ismatched = np.float32((name_resolve[1] == "match"))
            x = np.float32(name_resolve[2])
            y =  np.float32(name_resolve[3])
            w = np.float32(name_resolve[4])
            h = name_resolve[5]
            h = np.float32(h[0:len(h)-4])
            if ismatched:
                match_data[match_index, :] = [ismatched, x, y, w, h]
                match_index += 1
            else:
                unmatch_data[unmatch_index, :] = [ismatched, x, y, w, h]
                unmatch_index += 1
        self.match_data = match_data[0: match_index, :]
        self.unmatch_data = unmatch_data[0: unmatch_index, :]
        self.unmatch_index = unmatch_index

        self.draw_location_map()
        self.draw_size_map()
        self.draw_boundingbox()
        self.draw_missDistri()
        self.draw_sizeDistri()
        #self.draw_example()
        plt.show()





dataset = DataSet('0003')
dataset.generate_label()
dataset.display_analysis('')#260






