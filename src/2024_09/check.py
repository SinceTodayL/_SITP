import evaluate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Standard Label')
    plt.show()


def count_common_element(array1, array2):
    set_array1 = set(array1)
    set_array2 = set(array2)

    common_elements = set_array1.intersection(set_array2)

    return common_elements, len(common_elements)


def check_accuracy(self_detect):
    record_2 = [3, 6, 7, 74, 75, 78, 79, 80, 179, 183, 188, 195, 257, 265,
            268, 274, 387, 655, 730, 876, 894, 915, 924, 927, 998, 1001,
            1099, 1100, 1107, 1114, 1115, 1123, 1125]

    record_1 = [1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 54, 55,
            56, 58, 59, 60, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 76,
            77, 81, 82, 86, 95, 96, 97, 101, 102, 104, 105, 106, 107, 110, 112,
            114, 116, 118, 119, 120, 126, 127, 130, 134, 137, 140, 142, 150, 152,
            153, 154, 155, 156, 157, 159, 160, 161, 165, 167, 174, 175, 181, 184,
            185, 186, 187, 189, 191, 199, 201, 202, 203, 204, 205, 206, 207, 208,
            209, 211, 214, 216, 217, 219, 224, 225, 234, 238, 242, 248, 250, 252,
            253, 255, 256, 258, 259, 260, 261, 264, 267, 270, 271, 272, 273, 279,
            282, 284, 285, 287, 291, 293, 299, 302, 306, 307, 308, 319, 322, 327,
            328, 331, 332, 333, 336, 337, 339, 340, 342, 343, 346, 347, 352, 353,
            355, 358, 361, 364, 368, 370, 372, 374, 384, 388, 389, 392, 394, 395,
            397, 398, 400, 404, 415, 417, 418, 420, 422, 424, 426, 429, 433, 434,
            435, 436, 438, 439, 440, 441, 443, 444, 445, 454, 455, 456, 460, 463,
            467, 471, 472, 473, 474, 485, 501, 502, 503, 506, 507, 509, 511, 512,
            515, 523, 527, 528, 531, 535, 537, 551, 563, 570, 572, 578, 579, 583,
            586, 597, 599, 612, 615, 617, 619, 621, 623, 624, 628, 631, 633, 635,
            639, 656, 660, 661, 667, 673, 677, 684, 692, 693, 696, 698, 700, 707,
            709, 714, 720, 722, 724, 726, 729, 734, 736, 739, 749, 752, 754, 756,
            757, 758, 759, 760, 761, 763, 765, 766, 769, 776, 777, 784, 788, 789,
            790, 794, 798, 803, 806, 812, 818, 820, 821, 823, 824, 825, 826, 828,
            844, 845, 851, 852, 853, 854, 860, 861, 862, 863, 864, 868, 871, 872,
            877, 879, 881, 882, 883, 884, 885, 888, 889, 890, 892, 895, 896, 897,
            901, 906, 908, 909, 910, 911, 912, 913, 914, 916, 917, 918, 920, 923,
            933, 934, 938, 943, 944, 945, 949, 950, 956, 963, 967, 970, 972, 973,
            976, 979, 980, 983, 984, 987, 988, 990, 999, 1000, 1002, 1007, 1008,
            1009, 1010, 1011, 1014, 1017, 1024, 1027, 1032, 1033, 1034, 1035, 1036,
            1037, 1038, 1039, 1040, 1041, 1043, 1044, 1049, 1051, 1052, 1053, 1054,
            1056, 1057, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1068, 1069,
            1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1082,
            1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1098,
            1102, 1108, 1109, 1110, 1111, 1112, 1113, 1116, 1117, 1118, 1119, 1120,
            1121, 1122, 1124, 1126, 1127]

    index_intersection_record1, len_intersection_record1 = count_common_element(self_detect, record_1)
    index_intersection_record2, len_intersection_record2 = count_common_element(self_detect, record_2)

    file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\new_label.xlsx"
    df = pd.read_excel(file_path, header=None).to_numpy().flatten()
    record_1_array = []
    record_2_array = []

    for i in range(0, 1127):
        if df[i] == 0:
            record_1_array.append(1)
            record_2_array.append(1)
        elif df[i] == 1:
            record_1_array.append(-1)
            record_2_array.append(1)
        elif df[i] == 2:
            record_1_array.append(-1)
            record_2_array.append(-1)

    self_detect_ = [1]*1127
    for index in self_detect:
        self_detect_[index-1] = -1

    print("len of record_1: ", len(record_1))
    print("accuracy rate of record_1:", len_intersection_record1/len(self_detect))
    print("account for record_1:", len_intersection_record1/len(record_1))
    print("index of overlap: in record_1: ", index_intersection_record1)
    print()
    print("len of record_2: ", len(record_2))
    print("accuracy rate of record_2:", len_intersection_record2/len(self_detect))
    print("account for record_2:", len_intersection_record2/len(record_2))
    print("index of overlap: in record_2: ", index_intersection_record2)
    evaluate.evaluate_indicator(record_1_array, self_detect_)
    evaluate.evaluate_indicator(record_2_array, self_detect_)
    plot_confusion_matrix(record_1_array, self_detect_)
    plot_confusion_matrix(record_2_array, self_detect_)




