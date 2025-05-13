library(readxl)
library(ggplot2)
data_1 <- read_excel("C:/Users/LiuZh/Desktop/SITP/data_all.xlsx", range = "E1:G1126")
data_2 <- read_excel("C:/Users/LiuZh/Desktop/SITP/data_all.xlsx", range = "E1127:G2251")
# 查看读取的数据框的列名
colnames(data_1)
colnames(data_2)
ggplot(data_1, aes(x = BKM, y = value_left)) +
  geom_line() +
  labs(title = "Line Plot of Two Columns", x = "X Values", y = "Y Values") +
  theme_minimal()

ggplot(data_2, aes(x = BKM, y = value_left)) +
  geom_line() +
  labs(title = "Line Plot of Two Columns", x = "X Values", y = "Y Values") +
  theme_minimal()

ggsave("C:/Users/LiuZh/Desktop/SITP/plot.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)

# 创建分布图
ggplot(data_subset, aes(x = value_left, fill = "X1 Column")) + 
  geom_histogram(aes(y = after_stat(count)), bins = 30000, alpha = 0.5) +
  geom_density(alpha = 0.2) +
  labs(title = "Distribution of X1 Column Data", x = "X1 Column Value", y = "Density") +
  theme_minimal()




data_1 <- read_excel("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", range = "F1:H1128")
data_2 <- read_excel("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", range = "N1:P1128")

ggplot(data_1, aes(x = Girderindex, y = value_left)) +(data_2, aes(x = Girderindex, y = value_left))+
  geom_line() +
  labs(title = "Line Plot of Two Columns", x = "Girderindex", y = "value_left") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()

ggplot(data_2, aes(x = Girderindex, y = value_left)) +
  geom_line() +
  labs(title = "Line Plot of Two Columns", x = "Girderindex", y = "value_left") +
  
  theme_minimal()

library(ggplot2)
library(openxlsx)

# 读取数据
data_1 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 6:8)
data_2 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 14:16)

# 绘图
ggplot() +
  geom_line(data = data_1, aes(x = Girderindex, y = value_left), color = "blue") +
  geom_line(data = data_2, aes(x = Girderindex, y = value_left), color = "red") +
  labs(title = "Line Plot of Two Columns", x = "Girderindex", y = "value_left") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_1.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)