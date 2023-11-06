##### IMPORTS AND PACKAGES#####
install.packages("lme4")
library(lme4)
install.packages("lmerTest")
library(lmerTest)
install.packages("ggnewscale")
library(ggnewscale)
install.packages("tidyverse")
library(tidyverse)
install.packages("effects")
library(effects)
install.packages("gridExtra")
library(gridExtra)
install.packages("sjPlot")
library(sjPlot)
install.packages("ggcorrplot")
library(ggcorrplot)
install.packages("GGally")
library(GGally)
install.packages("lavaan")
library(lavaan)
install.packages("semPlot")
library(semPlot)


##### READ IN DATA #####
setwd("/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/day-to-day-modeling-files/")

obj_v_subj_dt <- read.csv("objective_subjective_merged_with_severity.csv", header = TRUE)

##### SET UP LMM #####

# filter out etiology = other rows. Only use this df if you are doing etiology analysis. 
obj_v_subj_dt_3 <- subset(obj_v_subj_dt, Etiology != "Other")

# impairment grouping
obj_v_subj_dt$Impaired <- factor(obj_v_subj_dt$Impaired, levels = c("Normal", "MCI", "Dementia"))

# mutate obj_v_subj_dt to add MoCA bins
obj_v_subj_dt <- obj_v_subj_dt %>%
  mutate(MoCA_binned = case_when(
    MoCA < 20 ~ "low(<20)",
    MoCA >= 20 & MoCA <= 25 ~ "intermediate(20-25)",
    MoCA > 25 ~ "high(>25)",
    TRUE ~ "unknown"  # Handling any other cases not covered by the conditions
  ))

# make sure that high MoCA is the default for regression comparisons
obj_v_subj_dt$MoCA_binned <- factor(obj_v_subj_dt$MoCA_binned, levels = c("high(>25)", "intermediate(20-25)", "low(<20)"))

##### DO THE PARTIAL POOLING #####
pp_mod <- lmer(Well.rested ~ WASO*MoCA_binned + Age + Sex + Etiology + (1 | Study.ID), data = obj_v_subj_dt) # For impairment model: change Etiology to Impaired, add etiology as a covariate, and change data to obj_v_subj_dt. For etiology, change data to obj_v_subj_dt, get rid of etiology as a covariate. 
# example syntax for impairment: 
#pp_mod <- lmer(`Well.rested` ~ `WASO`*Impaired + `Age` + `Sex` + Etiology +  (1 / `Study.ID`), data = obj_v_subj_dt)
# example syntax for etiology: 
#pp_mod <- lmer(Well.rested ~ WASO*Etiology + Age + Sex + (1 / Study.ID), data = obj_v_subj_dt_3)

# set up the model - QUADRATIC 
pp_mod_quad <- lmer(Overall.quality ~ poly(SleepDurationInSpt, 2, raw = TRUE)^2*Etiology + Age + Sex + (1 | Study.ID), data = obj_v_subj_dt_3)

# retrieve a summary of the model
pp_mod_summary <- summary(pp_mod)
pp_mod_summary_quad <-  summary(pp_mod_quad)

# get the partial pooling effects of etiology on objective variable, to plot. Whatever is in parenthesis should be what we are grouping by, and the objective variable. 
pp <- data.frame(Effect(c("MoCA_binned", "WASO"), pp_mod)) 

pp_quad <- data.frame(Effect(c("Etiology", "SleepDurationInSpt"), pp_mod_quad))


##### PLOTTING #####

# plot the model - MOCA, or ETIOLOGY
ggplot(pp, aes(x = WASO, y = fit, color = factor(MoCA_binned), group = MoCA_binned)) + 
  geom_line(size = 2) +
  geom_ribbon(aes(fill = factor(MoCA_binned), ymin = lower, ymax = upper), alpha = 0.1) + 
  labs(x = "Hours", y = "Well rested", color = "MoCA") +
  theme_classic(base_size = 23) + 
  scale_color_manual(name = "MoCA", values = c("low(<20)" = "#E41A1C", "intermediate(20-25)" = "#377EB8", "high(>25)" = "#4DAF4A")) +
  scale_fill_manual(name = "MoCA", values = c("low(<20)" = "#E41A1C", "intermediate(20-25)" = "#377EB8", "high(>25)" = "#4DAF4A"))

# plot the model - IMPAIRMENT
ggplot(pp, aes(x = WASO, y = fit, color = factor(Impaired), group = Impaired)) + 
  geom_line(size = 2) +
  geom_ribbon(aes(fill = factor(Impaired), ymin = lower, ymax = upper), alpha = 0.1) + 
  labs(x = "WASO", y = "Mentally Alert", color = "Impaired") +
  theme_classic(base_size = 20) + 
  scale_color_manual(name = "Impaired", values = c("Dementia" = "#E41A1C", "MCI" = "#377EB8", "Normal" = "#4DAF4A")) +
  scale_fill_manual(name = "Impaired", values = c("Dementia" = "#E41A1C", "MCI" = "#377EB8", "Normal" = "#4DAF4A"))


##### CORRELATION MATRIX AND VISUALIZATION #####

# set up the matrix
subset_data_for_cor_matrix <- obj_v_subj_dt_3 %>% select(Well.rested, Overall.quality, Deep.Sleep, Mentally.Alert)
cor <- round(cor(subset_data_for_cor_matrix, use = "pairwise.complete.obs"), 1)
# Create the ggcorrplot with a custom color scale from 0 to 1
ggcorrplot(cor, lab = TRUE, lab_size = 6) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 20),  # Adjust the text size for x-axis labels
    axis.text.y = element_text(size = 20),  # Adjust the text size for y-axis labels
    legend.text = element_text(size = 15),  # Adjust the legend text size
    legend.title = element_text(size = 16)  # Adjust the legend title size
  )


##### RESIDUALS, SLOPES ANALYSIS #####

# extract residuals from the model. 
filtered_data$WASO_res <- resid(lm(WASO~Age+Sex, data=filtered_data)) 
filtered_data$Well.rested_res <- resid(lm(Well.rested~Age+Sex, data=filtered_data)) 

# get a data frame with 112 columns, only having a unique data table 
unique_filtered_data <- obj_v_subj_dt_3 %>% distinct(Study.ID, .keep_all = TRUE)

### Extract and plot individual slopes ###
lme.model <- lmer(Well.rested ~ WASO + (WASO|Study.ID), data=obj_v_subj_dt_3)  
coef.model <- coef(lme.model)$Study.ID
# add the slopes to the filtered data df 
unique_filtered_data$slopes <- coef.model$WASO

# plot the histogram 
ggplot(unique_filtered_data, aes(x = slopes)) + 
  geom_histogram(color = "black", fill = "white", size =1) + 
  labs(x = "Slope", y = "Count", title = "Slope of WASO vs. Well rested") + 
  theme_classic(base_size = 20) +
  theme(plot.title = element_text(hjust = 0.5)) 

# create a box plot for each etiology, with etiology (HC, LB, AD) on the x-axis 
# and slope on the y-axis

ggplot(unique_filtered_data, aes(x = Etiology, y = slopes, color = factor(Etiology))) + 
  geom_boxplot(size = 1) + 
  geom_point() +
  theme_classic(base_size = 20) + 
  labs(x = 'Etiology', y = 'Slope', title = 'WASO vs. Well rested', color = "Etiology") + 
  theme(plot.title = element_text(hjust = 0.5)) 


##### PLOT BOX PLOT OF BASELINE MEASURES ##### 

# filter unique_filtered_data so it has a mean value for WASO for EACH USER
# Group by 'Study.ID' and calculate the mean of 'WASO'
df_one_mean_subj_per_ID <- obj_v_subj_dt_3 %>%
  group_by(Study.ID) %>%
  summarise(mean_sleep_duration = mean(SleepDurationInSpt, na.rm = TRUE))

result_df <- right_join(df_one_mean_subj_per_ID, unique_filtered_data, by = "Study.ID")


# Make a box plot of each objective and subjective measure, by etiology 
ggplot(result_df, aes(x = Etiology, y = WASO, color = factor(Etiology))) + 
  geom_boxplot(size = 1) + 
  scale_color_manual(name = "Etiology", values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) + 
  theme_classic(base_size = 20) + 
  labs(x = 'Etiology', y = 'WASO (hours)', color = "Etiology")  
  

# calculate the mean by 'Study.ID' first
mean_by_study <- aggregate(obj_v_subj_dt_3[, c("Well.rested", "Deep.Sleep", "Mentally.Alert", "Overall.quality")],
                           by = list(obj_v_subj_dt_3$Study.ID),
                           mean,
                           na.rm = TRUE)

# Rename the "Group.1" column to "Study.ID"
colnames(mean_by_study)[1] <- "Study.ID"

# merge the dfs, based on Study ID
merged_data <- merge(unique_filtered_data, mean_by_study, by = "Study.ID")

# Now, calculate the mean by 'Etiology'
mean_data <- aggregate(mean_by_study[, 2:5],
                       by = list(merged_data$Etiology),
                       mean,
                       na.rm = TRUE)


# Convert the mean_data to a long format for plotting
mean_data_long <- tidyr::gather(mean_data, key = "SubjectiveMeasure", value = "MeanValue", -Group.1)



# Custom labels for the subjective measures
custom_labels <- c("Mentally Alert" = "Mentally.Alert",
                   "Deep Sleep" = "Deep.Sleep",
                   "Overall Quality" = "Overall.quality",
                   "Well Rested" = "Well.rested")

# Plot the box plot with grouped bars and labels for the subjective measures
# weird plot debug: rename the y tick marks manually
# Define the new labels for y-axis ticks
new_y_labels <- c(
  "0" = "1",
  "1" = "2",
  "2" = "3",
  "3" = "4",
  "4" = "5", 
  "5" = "6"
)


ggplot(mean_data_long, aes(x = SubjectiveMeasure, y = MeanValue, fill = factor(Group.1))) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(data = subset(mean_data_long, !duplicated(SubjectiveMeasure)), 
            aes(label = names(custom_labels), y = max(MeanValue) + 0.5),
            position = position_dodge(width = 0.8),
            vjust = -0.5,
            size = 8) +
  scale_x_discrete(labels = custom_labels) +
  scale_y_continuous(limits = c(0, 5), expand = c(0, 0)) +  # Set y-axis limits from 0 to 5 
  coord_cartesian(
    #xlim = NULL,
    ylim = c(1, 5),
    expand = TRUE,
    default = FALSE,
    clip = "on"
  ) + 
  theme_classic(base_size = 24) +
  labs(x = "", y = "Mean Score", fill = "Etiology") +
  scale_fill_manual(values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  theme(
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.border = element_blank())

 
out_of_range_rows <- which(mean_data_long$MeanValue < 1 | mean_data_long$MeanValue > 5)
print(mean_data_long[out_of_range_rows, ])


########## CONFIRMATORY FACTOR ANALYSIS ##########
# set up model
CFA.model <- 'subjective sleep quality  =~ Mentally.Alert + Deep.Sleep + Overall.quality + Well.rested'

# remove any missing data from obj_v_subj_dt 


# fit model to our data
fit <- cfa(CFA.model, data = obj_v_subj_dt)

# view the summary of the model
summary(fit, fit.measures = TRUE, standardized = T)

semPaths(fit, "std", edge.label.cex = 0.5, curvePivot = TRUE)

# add this as column (remove na's from dt first)
obj_v_subj_dt$latent_variable <- predict(fit)





