t <- read.table("JS77_filtered_2.txt", sep = "\t", header = T)

idx <- t$Group %in% c("LT_Day56", "ST_Day28", "MPP_Day14", "ProDC_Day9", "CMP_Day9", "CLP_Day14")

write.table(t[idx, ], "JS77_filtered_2_first_timepoint.txt", row.names = F)
write.table(t[!idx, ], "JS77_filtered_2_second_timepoint.txt", row.names = F)
