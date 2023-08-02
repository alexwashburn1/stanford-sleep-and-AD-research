##### IMPORTS #####
install.packages("lme4")
library(lme4)

library(lmerTest)

install.packages("ggnewscale")
library(ggnewscale)


install.packages("tidyverse")
library(tidyverse)

install.packages("effects")
library(effects)
install.packages("gridExtra")
library(gridExtra)


##### READ IN DATA #####
setwd("/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/day-to-day-modeling-files/")

obj_v_subj_dt <- read.csv("objective_subjective_merged.csv", header = TRUE)

print(head(obj_v_subj_dt))
print(colnames(obj_v_subj_dt))


##### SET UP LMM #####

# partial pooling model
#pp_mod <- lmer(`Well.rested` ~ `WASO` + `Age` + `Sex` + (`WASO` / `Study.ID`), obj_v_subj_dt)

obj_v_subj_dt_3 <- subset(obj_v_subj_dt, Etiology != "Other")
obj_v_subj_dt_3$Etiology <- factor(obj_v_subj_dt_3$Etiology, levels = c("HC", "AD", "LB"))

# set up the model
pp_mod <- lmer(`Well.rested` ~ sleep_efficiency*Etiology + Age + Sex + (1 | Study.ID), data = obj_v_subj_dt_3)

# retrieve a summary of the model
summary(pp_mod)

# get the partial pooling effects of etiology on objective variable, to plot
pp <- data.frame(Effect(c("Etiology", "sleep_efficiency"), pp_mod))

ggplot(pp, aes(x = sleep_efficiency, y = fit, color = Etiology, group = Etiology)) + 
  geom_line(size = 1.5) +# geom_ribbon(aes(color = NA, fill = factor(Etiology),  
  #ymin = lower, ymax = upper), alpha = 0.3) +   
  labs(x = "Sleep Efficiency", y = "Well Rested") +
  theme_classic(base_size=20) + 
  scale_color_manual(values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A"))


# extract residuals from the model. 
filtered_data$WASO_res <- resid(lm(WASO~Age+Sex, data=filtered_data)) 
filtered_data$Well.rested_res <- resid(lm(Well.rested~Age+Sex, data=filtered_data)) 


# run the prediction analysis
newdata <- obj_v_subj_dt %>%
  mutate(Well.rested = predict(pp_mod, obj_v_subj_dt))

print(head(newdata))

# extract only subject, WASO, predicted_well_restedness 
newdata2 <- newdata %>% 
  select(Study.ID, WASO, Well.rested)

# Filter out rows with etiology = "Other"
filtered_data <- obj_v_subj_dt %>% filter(Etiology != "Other")

# Assuming the column containing the number of measurements is called "MeasurementCount"
etiology_counts <- filtered_data %>%
  group_by(Etiology) %>%
  summarise(MeasurementCount = n())

# Add the text labels to the plot using annotate
ggplot(filtered_data, aes(x = WASO_res, y = Well.rested_res, color = Etiology)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  theme_classic(base_size = 20) +
  labs(x = "WASO", y = "Well Rested", color = "Etiology") +
  # Add the text labels in the plot legend
  annotate("text", x = Inf, y = -Inf, label = paste(etiology_counts$Etiology, " (n =", etiology_counts$MeasurementCount, ")"),
           hjust = 1, vjust = 0, color = "black", size = 4) +
  guides(color = guide_legend(override.aes = list(size = 4)))

View(newdata2)


