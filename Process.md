## Process



2025 0320

对 `07_2020-10-23_lev/gui` 数据集作数据填充（平均值）：

`07_2020-10-23_lev`  ：2/1127/1128 line: value_left, value_right, label(0)

`07_2020-10-23_gui` ： 2/1128 line: value_left, value_right, label(0)



2025 0322

画出来几张波形图，比较这么多年波形的重合度和走势，发现 $1,6,7,9,10,11$ 这些标号的文件离群较严重 



>时空连续性：轨道数据具有时间特性，本身也具有空间连续性

引入回归指数，用transformer模型连续预测未来10年的值
$$
RI_j = \sum_{i = 1}^{years} |\sum_{i=1}^{4}(X_{i,j,k} - X_{i-1,j,k})|
$$
得到异常大的值如下：

Sample index 2: Regression Index = 161.03610229492188
Sample index 18: Regression Index = 107.47587585449219
Sample index 34: Regression Index = 80.59025573730469
Sample index 46: Regression Index = 86.84085083007812
Sample index 63: Regression Index = 91.65447998046875
Sample index 65: Regression Index = 85.34539031982422
Sample index 71: Regression Index = 84.99043273925781
Sample index 73: Regression Index = 106.25669860839844
Sample index 77: Regression Index = 84.89530944824219
Sample index 79: Regression Index = 86.49015045166016
Sample index 89: Regression Index = 86.40185546875
Sample index 98: Regression Index = 82.0219497680664
Sample index 126: Regression Index = 99.77745819091797
Sample index 256: Regression Index = 96.81896209716797
Sample index 269: Regression Index = 85.10673522949219
Sample index 298: Regression Index = 80.15121459960938
Sample index 299: Regression Index = 96.70356750488281
Sample index 311: Regression Index = 99.74298095703125
Sample index 386: Regression Index = 88.10344696044922
Sample index 398: Regression Index = 86.60328674316406
Sample index 433: Regression Index = 91.14056396484375
Sample index 443: Regression Index = 78.57553100585938
Sample index 444: Regression Index = 108.47859954833984
Sample index 449: Regression Index = 82.04297637939453
Sample index 453: Regression Index = 86.09656524658203
Sample index 484: Regression Index = 95.9910888671875
Sample index 527: Regression Index = 91.06782531738281
Sample index 577: Regression Index = 79.98064422607422
Sample index 586: Regression Index = 81.72273254394531
Sample index 733: Regression Index = 78.21369934082031
Sample index 827: Regression Index = 90.45085144042969
Sample index 840: Regression Index = 78.98478698730469
Sample index 857: Regression Index = 78.92745208740234
Sample index 863: Regression Index = 85.44541931152344
Sample index 903: Regression Index = 83.41056060791016
Sample index 911: Regression Index = 95.57362365722656
Sample index 917: Regression Index = 111.17932891845703
Sample index 968: Regression Index = 87.01858520507812
Sample index 1034: Regression Index = 94.76737976074219
Sample index 1045: Regression Index = 95.29264831542969
Sample index 1047: Regression Index = 93.61658477783203
Sample index 1076: Regression Index = 110.8653793334961
Sample index 1079: Regression Index = 136.20556640625
Sample index 1088: Regression Index = 102.89595031738281
Sample index 1107: Regression Index = 106.49848175048828
Sample index 1108: Regression Index = 86.79864501953125
Sample index 1109: Regression Index = 85.7473373413086
Sample index 1112: Regression Index = 156.04135131835938
Sample index 1117: Regression Index = 138.8775634765625
Sample index 1120: Regression Index = 91.7652359008789
Sample index 1121: Regression Index = 93.90483856201172

但这种方法的合理性存疑，这计算的只是一种“波动指数”, 不能用来衡量异常
