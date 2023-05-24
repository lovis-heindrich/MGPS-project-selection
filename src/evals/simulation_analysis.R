library(emmeans)

# Read data
df = read.csv("./data/simulation_results/simulation_data.csv")
df$Name <- as.factor(df$type)

# ANOVA model
model <- aov(expected_reward ~ type, data=df)
summary(model)

# Test main effect of the used algorithm
TukeyHSD(aov(expected_reward ~ type, data=df), "type")
