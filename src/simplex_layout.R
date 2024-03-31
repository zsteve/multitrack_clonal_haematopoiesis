library(plotly)

df <- read.csv("JS77_filtered_2.txt", sep = "\t")
X <- df[names(df) %in% c("cDC1", "cDC2", "pDC", "Eos", "Mon", "Neu", "B")]
X <- X[rowSums(X) > 5, ]
X_log <- log1p(X)
X_log_norm <- X_log / rowSums(X_log)

S <- cor(X_log)

heatmap(S)

y <- data.frame(cmdscale(1 - S^2, k = 3))
names(y) <- c("MDS1", "MDS2", "MDS3")

X_simplex <- data.frame(as.matrix(X_log_norm) %*% as.matrix(y))
df <- merge(df, X_simplex, by = 0)

p <- plot_ly(data = df, x = ~MDS1, y = ~MDS2, z = ~MDS3, type = "scatter3d", color = ~Day, opacity = 0.25, hoverinfo = text, text = paste(df$fate, df$Group))
p <- add_text(p, x = y$MDS1, y = y$MDS2, z = y$MDS3, text = rownames(y), name = "Celltypes", color = I("black"))
p

