#2020_value_right
# 读取数据
data_1 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 62:64)

# 绘图
ggplot() +
  geom_line(data = data_1, aes(x = Girderindex, y = value_right), color = "blue") +
  
  labs(title = "2020_value_lev_right", x = "Girderindex", y = "value_right") +
  
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_2020_lev_right.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)

data_2 <- read.xlsx("C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rows = 1:1128, cols = 86:88)

#2022_value_left
# 绘图
ggplot() +
  geom_line(data = data_2, aes(x = Girderindex, y = value_right), color = "blue") +
  
  labs(title = "2022_value_gui_left", x = "Girderindex", y = "value_left") +
  
  theme_minimal()
ggsave("C:/Users/LiuZh/Desktop/SITP/plot_2022_gui_left.png", plot = last_plot(), width = 20, height = 4, dpi = 1000)

