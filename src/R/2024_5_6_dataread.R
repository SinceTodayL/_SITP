library(readr)
library(openxlsx)

folder_path <- "C:/Users/LiuZh/Desktop/SITP/data"
#这是我的data文件所在位置
txt_files <- list.files(folder_path, pattern = "\\.txt$", full.names = TRUE)

all_data <- data.frame()

#遍历文件夹
for (file in txt_files) 
{
  current_data <- read_delim(file, delim = "\t", col_names = FALSE)
#以“_”为标志分割文件名
  file_name_parts <- unlist(strsplit(basename(file), "_"))
  
  Date <- paste(file_name_parts [1], collapse = "_")
  Time <- paste(file_name_parts [2], collapse = "_")
  Type <- paste(file_name_parts[3], collapse = "_")
  GUI  <- paste(file_name_parts[4], collapse = "_")

 # 创建与数据框行数相同的新列，并将提取的内容填充到新列中
  Date_col <- rep(Date, nrow(current_data))
  Time_col <- rep(Time, nrow(current_data))
  Type_col <- rep(Type, nrow(current_data))
  GUI_col  <- rep(GUI, nrow(current_data))

 # 将提取的内容作为新列添加到原数据中
  current_data <- cbind(Date_col,Time_col,Type_col,GUI_col, current_data)
  all_data <- rbind(all_data,current_data)
}

colnames(all_data) <- c( "Date", "Time", "Type", "Gui","BKM" ,"Girderindex","value_left","value_right")
write.xlsx(all_data, file ="C:/Users/LiuZh/Desktop/SITP/data_all.xlsx", rowNames = FALSE)




#下面的代码是把每一年的数据合并到一起

library(readr)
library(openxlsx)

folder_path <- "C:/Users/LiuZh/Desktop/SITP/data_copy"
#这是我的data文件所在位置
txt_files <- list.files(folder_path, pattern = "\\.txt$", full.names = TRUE)

all_data <- data.frame( )

#遍历文件夹
for (file in txt_files) 
{
  current_data <- read_delim(file, delim = "\t", col_names = FALSE)
  #以“_”为标志分割文件名
  file_name_parts <- unlist(strsplit(basename(file), "_"))
  
  Date <- paste(file_name_parts [1], collapse = "_")
  Time <- paste(file_name_parts [2], collapse = "_")
  Type <- paste(file_name_parts[3], collapse = "_")
  GUI  <- paste(file_name_parts[4], collapse = "_")
  
  # 创建与数据框行数相同的新列，并将提取的内容填充到新列中
  Date_col <- rep(Date, nrow(current_data))
  Time_col <- rep(Time, nrow(current_data))
  Type_col <- rep(Type, nrow(current_data))
  GUI_col  <- rep(GUI, nrow(current_data))
  
  # 将提取的内容作为新列添加到原数据中
  current_data <- cbind(Date_col,Time_col,Type_col,GUI_col, current_data)
  all_data <-cbind(all_data,current_data)
}
all_colnames <- c()
col_names <-c( "Date", "Time", "Type", "Gui","BKM" ,"Girderindex","value_left","value_right")
for(i in 1:14){
  all_colnames <-c(all_colnames,col_names)
}
colnames(all_data) <- all_colnames
write.xlsx(all_data, file = "C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rowNames = FALSE)



#GPT
library(readr)
library(openxlsx)

folder_path <- "C:/Users/LiuZh/Desktop/SITP/data_copy"
txt_files <- list.files(folder_path, pattern = "\\.txt$", full.names = TRUE)

all_data <- NULL  # 创建一个空的数据框

# 遍历文件夹
for (file in txt_files) {
  current_data <- read_delim(file, delim = "\t", col_names = FALSE)
  # 以“_”为标志分割文件名
  file_name_parts <- unlist(strsplit(basename(file), "_"))
  
  Date <- paste(file_name_parts[1], collapse = "_")
  Time <- paste(file_name_parts[2], collapse = "_")
  Type <- paste(file_name_parts[3], collapse = "_")
  GUI <- paste(file_name_parts[4], collapse = "_")
  
  # 添加新列
  current_data <- cbind(Date = Date, Time = Time, Type = Type, GUI = GUI, current_data)
  
  if (is.null(all_data)) {
    all_data <- current_data  # 第一次循环直接赋值
  } else {
    all_data <- cbind(all_data, current_data)  # 从第二次循环开始逐个添加列
  }
}

# 设置列名
col_names <- c("Date", "Time", "Type", "Gui", "BKM", "Girderindex", "value_left", "value_right")
colnames(all_data) <- col_names

# 将数据写入 Excel 文件
write.xlsx(all_data, file = "C:/Users/LiuZh/Desktop/SITP/data_copy_combine.xlsx", rowNames = FALSE)

