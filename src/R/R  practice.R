matrix_example <- matrix(
  c(20:45,1:10),
  nrow=6,
  #ncol=6  
  byrow=T,
  #byrow = F by default, should be changed
  dimname=list(c('col1','col2','col3','col4','col5','col6'),c('row1','row2','row3','row4','row5','row6'))
)
matrix_example
det(matrix_example)
a <- matrix_example[matrix_example%%3==0]
b<- matrix_example[c('col1','col6'),c('row3','row1')]
a2 <- rank(matrix_example)
svd_matrix_example=svd(matrix_example)
a3 <- rank(matrix)


# 读取数据
data <- read.table("C:/Users/LiuZh/Desktop/SITP：面向磁浮轨道异常检测的大数据分析/data/2017-11-01_10-43_PA_gui_GMS-Longwave_Tab.txt", header = TRUE)  # 请将 "your_data_file.txt" 替换为你的数据文件路径

# 检查数据结构
str(data)

# 计算相关系数
correlation_matrix <- cor(data)

# 打印相关系数矩阵
print(correlation_matrix)

# 可视化相关系数矩阵（如果你有兴趣）
# 你可以使用 corrplot 包或其他可视化工具
# install.packages("corrplot")  # 如果你没有安装 corrplot 包，请取消注释此行并运行
# library(corrplot)
# corrplot(correlation_matrix, method="circle")

# 可以进一步分析相关性，如绘制散点图、回归分析等，具体取决于你的需求

data <- read.table("C:/Users/LiuZh/Desktop/SITP/2024_4_21/2017-11-01_10-43_PA_gui_GMS-Longwave_Tab.txt",header=TRUE)
data_sub <- data[,c(3,4)]
colnames(data) <- c("col1","col2","col3","col4")
correlation_data_2017_1 <- cor(data)
print(correlation_data_2017_1)
colnames(data_sub) <- c("value_left","value_right")
correlation_data_sub_2017_1 <- cor(data_sub)
print(correlation_data_sub_2017_1)


data <- read.table("C:/Users/LiuZh/Desktop/SITP/2024_4_21/2017-11-01_10-49_PA_lev_GMS-Longwave_Tab.txt",header=TRUE)
data_sub <- data[,c(3,4)]
colnames(data) <- c("col1","col2","col3","col4")
correlation_data_2017_1 <- cor(data)
print(correlation_data_2017_1)
colnames(data_sub) <- c("value_left","value_right")
correlation_data_sub_2017_1 <- cor(data_sub)
print(correlation_data_sub_2017_1)

data <- read.table("C:/Users/LiuZh/Desktop/SITP/2024_4_21/2018-11-16_08-59_LA_gui_GMS-Longwave_Tab.txt",header=TRUE)
data_sub <- data[,c(3,4)]
colnames(data) <- c("col1","col2","col3","col4")
correlation_data_2018_1 <- cor(data)
print(correlation_data_2017_1)
colnames(data_sub) <- c("value_left","value_right")
correlation_data_sub_2017_1 <- cor(data_sub)
print(correlation_data_sub_2018_1)

# 加载 ggplot2 包
library(ggplot2)

data <- read.table("C:/Users/LiuZh/Desktop/SITP/2024_4_21/2018-11-16_08-59_LA_gui_GMS-Longwave_Tab.txt",header=TRUE)
data_sub <- data[,c(3,4)]
colnames(data_sub) <- c("value_left","value_right")

scatter_plot <- ggplot(data_sub, aes(x = value_left, y = value_right)) +
geom_point() +  
geom_smooth(method = "lm", se = FALSE) +
labs(x = "value_left", y = "value_right", title = "relation between value_L and value_R 2018_1")  # 添加标签和标题

print(scatter_plot)
ggsave("C:/Users/LiuZh/Desktop/SITP/2024_4_21/relation between value_L and value_R 2018_1.png",plot=scatter_plot)

# 加载 ggplot2 包
library(ggplot2)
data <- read.table("C:/Users/LiuZh/Desktop/SITP/2024_4_21/2018-11-16_09-00_LA_lev_GMS-Longwave_Tab.txt",header=TRUE)
data_sub <- data[,c(3,4)]
colnames(data_sub) <- c("value_left","value_right")
# 创建散点图
scatter_plot <- ggplot(data_sub, aes(x = value_left, y = value_right)) +
  geom_point() +  # 添加散点
  #geom_smooth(method = "lm", se = FALSE) +  # 添加线性拟合线，不显示标准误差带http://127.0.0.1:26065/graphics/ba791b00-3923-4ad6-8fe9-b19d110668a1.png
  labs(x = "value_left", y = "value_right", title = "relation between value_L and value_R 2018_2")  # 添加标签和标题

# 显示散点图
print(scatter_plot)
ggsave("C:/Users/LiuZh/Desktop/SITP/2024_4_21/relation between value_L and value_R 2018_2.png",plot=scatter_plot)



library(ggplot2)
data <- read.table("C:/Users/LiuZh/Desktop/SITP/2024_4_21/2018-11-16_08-59_LA_gui_GMS-Longwave_Tab.txt",header=TRUE)
data_sub <- data[,c(2,3,4)]
colnames(data_sub) <- c("seq","value_left","value_right")

lm_model <- lm(value_left ~ value_right, data = data_sub)

intercept <- coef(lm_model)[1]
slope <- coef(lm_model)[2]

residuals <- resid(lm_model)

residual_plot <- ggplot(data_sub, aes(x = seq, y = residuals)) +
  geom_point() +  # 添加残差散点
  geom_abline(intercept = 0, slope = 0, linetype = "dashed", color = "blue") +  # 添加 y = 0 的参考线
  geom_abline(intercept = -1, slope = 0, linetype = "dashed", color = "red")+
  geom_abline(intercept = +1, slope = 0, linetype = "dashed", color = "red")+
  labs(x = "Data_seq", y = "Residuals", title = "2018_1 Residual Plot")  # 添加标签和标题

print(residual_plot)

# 保存残差图为 PNG 文件
#ggsave("residual_plot.png", plot = residual_plot, width = 6, height = 4, dpi = 300)