
library(ggplot2)
library(openxlsx)

#2017_value_left
# 读取数据
data_1 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 6:8)
data_2 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 14:16)

# 绘图
ggplot() +
  geom_line(data = data_1, aes(x = Girderindex, y = value_left), color = "blue") +
  geom_line(data = data_2, aes(x = Girderindex, y = value_left), color = "red") +
  labs(title = "2017_value_left", x = "Girderindex", y = "value_left") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_2017_left.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)




#2017_value_right
# 读取数据
data_1 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 6:8)
data_2 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 14:16)

# 绘图
ggplot() +
  geom_line(data = data_1, aes(x = Girderindex, y = value_right), color = "blue") +
  geom_line(data = data_2, aes(x = Girderindex, y = value_right), color = "red") +
  labs(title = "2017_value_right", x = "Girderindex", y = "value_left") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_2017_right.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)


#2018_value_left
# 读取数据
data_1 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 22:24)
data_2 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 30:32)

# 绘图
ggplot() +
  geom_line(data = data_1, aes(x = Girderindex, y = value_left), color = "blue") +
  geom_line(data = data_2, aes(x = Girderindex, y = value_left), color = "red") +
  labs(title = "2018_value_left", x = "Girderindex", y = "value_left") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_2018_left.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)




#2018_value_right
# 读取数据
data_1 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 22:24)
data_2 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 30:32)

# 绘图
ggplot() +
  geom_line(data = data_1, aes(x = Girderindex, y = value_right), color = "blue") +
  geom_line(data = data_2, aes(x = Girderindex, y = value_right), color = "red") +
  labs(title = "2018_value_right", x = "Girderindex", y = "value_left") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_2018_right.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)




#2019_value_left
# 读取数据
data_1 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 38:40)
data_2 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 46:48)

# 绘图
ggplot() +
  geom_line(data = data_1, aes(x = Girderindex, y = value_left), color = "blue") +
  geom_line(data = data_2, aes(x = Girderindex, y = value_left), color = "red") +
  labs(title = "2019_value_left", x = "Girderindex", y = "value_left") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_2019_left.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)




#2019_value_right
# 读取数据
data_1 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 38:40)
data_2 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 46:48)

# 绘图
ggplot() +
  geom_line(data = data_1, aes(x = Girderindex, y = value_right), color = "blue") +
  geom_line(data = data_2, aes(x = Girderindex, y = value_right), color = "red") +
  labs(title = "2019_value_right", x = "Girderindex", y = "value_left") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_2019_right.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)
