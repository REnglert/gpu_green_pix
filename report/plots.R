library(data.table)
library(dplyr)
library(ggplot2)

data <- fread("data2.csv") %>% mutate(accuracy = recorded / real)

lm1 <- lm(recorded ~ real, data = data)
summary(lm1)

ggplot(data, aes(x = real, y = recorded)) + 
    geom_point(size = 2, aes(color=factor(num_per_thread))) + 
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "black") + 
    annotate("text", x=950, y = 350, label = "R^2 = 0.9771", size = 6) + 
    labs(x = "Real Leaf Area (cm^2)", y = "GPIX-calculated Leaf Area (cm^2)", title = "Pixel Counting: Real vs Calculated Leaf Area ") + 
    theme_bw() + theme(legend.position = "none")

ggsave("plot1.png", width = 6, height = 4, units = "in")

sum <- data %>% group_by(num_per_thread) %>% summarise(mean_runtime = mean(runtime), mean_accuracy = mean(accuracy), sd = sd(accuracy))

ggplot(sum, aes(x=mean_runtime, y = mean_accuracy)) + 
    geom_errorbar(width = 0.005, aes(x = mean_runtime, y = mean_accuracy, ymin = mean_accuracy-sd, ymax = mean_accuracy+sd)) + 
    geom_point(aes(color=factor(num_per_thread)), size = 3) + 
    coord_cartesian(ylim = c(0, 1), xlim = c(0.178, 0.322), expand = FALSE) + 
    annotate("text", x=sum$mean_runtime, y = c(.6, .6, .6, .6), 
             label = c("1 px", "4 px", "9 px", "16 px"), size = 6) + 
    labs(x = "Kernel Runtime (ms)", y = "Accuracy (Real / GPIX)", title = "Pixel Counting: Runtime vs Accuracy, Pixels counted per thread") + 
    theme_bw() + theme(legend.position = "none")

ggsave("plot2.png", width = 6, height = 4, units = "in")

sobel_par = 0.03801
sobel_seq = 14.226688
canny_par = 0.78202
laplacian_par= 0.751840
gcount_par = 0.503200
gcount_seq = 16.578465

Version = c("Sobel Parallel", "Sobel Sequential","Canny Parallel", "Laplacian Parallel", "Pixel Count Parallel", "Pixel Count Sequential")
name = c("Sobel", "Sobel",  "Canny", "Laplacian", "Pixel Count", "Pixel Count")
runtime = c(sobel_par, sobel_seq, canny_par, laplacian_par, gcount_par, gcount_seq)

barData <- tibble(name, Version, runtime)

ggplot(barData, aes(x = name, y = runtime)) + 
    geom_bar(stat = "identity", aes(color = Version, fill = Version), position = position_dodge(), alpha = 0.8, size = 1.25) + 
    labs(x = "Kernel Name", y = "Runtime (ms)", title = "Parallel vs Sequential Performance") + 
    coord_cartesian(ylim = c(0, 18)) + 
    annotate("text", x = c(3, 4), y = c(gcount_seq, sobel_seq)+1, label = c("33x","355x"), size = 5) + 
    theme_bw()

ggsave("plot3.png", width = 6, height = 4, units = "in")

