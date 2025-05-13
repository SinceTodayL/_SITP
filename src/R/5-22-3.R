library(ggplot2)
library(openxlsx)

#2017-2018_valule_left
# 读取数据
data_1 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 6:8)
data_2 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 22:24)

# 绘图
ggplot() +
  geom_line(data = data_1, aes(x = Girderindex, y = value_left), color = "blue") +
  geom_line(data = data_2, aes(x = Girderindex, y = value_left), color = "red") +
  labs(title = "2017-2018_value_left", x = "Girderindex", y = "value_left") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_2017-2018_left.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)


#2022-2023_valule_left
# 读取数据
data_1 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 86:88)
data_2 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 102:104)

# 绘图
ggplot() +
  geom_line(data = data_1, aes(x = Girderindex, y = value_left), color = "blue") +
  geom_line(data = data_2, aes(x = Girderindex, y = value_left), color = "red") +
  labs(title = "2022-2023_valule_left", x = "Girderindex", y = "value_left") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_2022-2023_valule_left.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)
